import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, t

data = pd.read_csv('EURUSD30.csv', header=None)
data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'])

# Выбор цен открытия с интервалом 180 минут (3 часа)
data_180 = data.iloc[::3]

data_180['Log_Returns'] = np.log(data_180['Open'] / data_180['Open'].shift(1))

data_180 = data_180.dropna()

def compute_autocorrelations(series, max_lag, method):
    autocorrelations = []
    p_values = []
    critical_values = []  # Критические значения для каждого лага
    alpha = 0.1  # Уровень значимости
    for lag in range(1, max_lag + 1):
        if len(series[lag:]) < 2 or len(series[:-lag]) < 2:
            break  
        n = len(series[lag:])
        df = n - 2  # Степени свободы
        if method == 'pearson':
            corr, p = pearsonr(series[lag:], series[:-lag])
        elif method == 'spearman':
            corr, p = spearmanr(series[lag:], series[:-lag])
        elif method == 'kendall':
            corr, p = kendalltau(series[lag:], series[:-lag])

        t_critical = t.ppf(1 - alpha / 2, df=df)  
        critical_value = t_critical / np.sqrt(df + t_critical**2)

        autocorrelations.append(corr)
        p_values.append(p)
        critical_values.append(critical_value)
    return autocorrelations, p_values, critical_values


max_lag = len(data_180) - 20 

pearson_autocorr, pearson_p, pearson_critical = compute_autocorrelations(data_180['Log_Returns'], max_lag, 'pearson')
spearman_autocorr, spearman_p, spearman_critical = compute_autocorrelations(data_180['Log_Returns'], max_lag, 'spearman')
kendall_autocorr, kendall_p, kendall_critical = compute_autocorrelations(data_180['Log_Returns'], max_lag, 'kendall')


def plot_autocorrelogram(autocorrelations, p_values, critical_values, title):
    lags = range(1, len(autocorrelations) + 1)  # Lag начинается с 1
    significant_pos = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if corr > crit]
    significant_neg = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if corr < -crit]
    insignificant = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if -crit <= corr <= crit]

    plt.figure(figsize=(10, 6))
    plt.scatter(significant_pos, [autocorrelations[lag - 1] for lag in significant_pos], color='blue', label='Положительные корреляции', s=10)
    plt.scatter(significant_neg, [autocorrelations[lag - 1] for lag in significant_neg], color='red', label='Отрицательные корреляции', s=10)
    plt.scatter(insignificant, [autocorrelations[lag - 1] for lag in insignificant], color='black', label='Незначимые корреляции', s=10)


    print(critical_values)
    negative_critical=[]
    for i in critical_values:
      negative_critical.append(i*-1)
    plt.plot(critical_values,color='green')
    plt.plot(negative_critical,color='orange')

    plt.axhline(0, color='black', linewidth=0.5)

    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Корреляция')
    plt.legend()
    plt.show()



plot_autocorrelogram(pearson_autocorr, pearson_p, pearson_critical, 'Пирсон')
plot_autocorrelogram(spearman_autocorr, spearman_p, spearman_critical, 'Спирмен')
plot_autocorrelogram(kendall_autocorr, kendall_p, kendall_critical, 'Кендалл')