
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr, kendalltau, t
from scipy.stats import chatterjeexi

# Загрузка данных
data = pd.read_csv('USDRUR60.csv', header=None)
data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Преобразование даты и времени в формат datetime
data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'])

# Выбор цен открытия с интервалом 150 минут 
data_150 = data.iloc[::4]

# Вычисление логарифмических доходностей
data_150['Log_Returns'] = np.log(data_150['Open'] / data_150['Open'].shift(1))

# Удаление первой строки с NaN
data_150 = data_150.dropna()


data_150['new']=data_150['Open']


# Функция для вычисления автокорреляций
def compute_autocorrelations(series, max_lag, method):
    autocorrelations = []
    p_values = []
    critical_values = []  # Критические значения для каждого лага
    alpha = 0.1  # Уровень значимости
    for lag in range(1, max_lag + 1):
        if len(series[lag:]) < 2 or len(series[:-lag]) < 2:
            break  # Прекращаем вычисления, если данных недостаточно
        n = len(series[lag:])  # Количество данных для текущего лага
        df = n - 2  # Степени свободы
        corr, p = chatterjeexi(series[lag:], series[:-lag])

        # Вычисление критического значения корреляции для уровня значимости alpha
        critical_value = norm.ppf(1 - alpha/2) * np.sqrt((len(series) + 3) / (9 * len(series) * (len(series) - 1)))

        autocorrelations.append(corr)
        p_values.append(p)
        critical_values.append(critical_value)
    return autocorrelations, p_values, critical_values

# Параметры
max_lag = 50  # Максимальный lag равен половине длины ряда




# Вычисление автокорреляций для каждого метода
ch_autocorr, ch_p, ch_critical = compute_autocorrelations(data_150['Log_Returns'], max_lag, 'pearson')
print(data_150['Log_Returns'])
newch_autocorr, newch_p, newch_critical = compute_autocorrelations(data_150['new'], max_lag, 'pearson')




# Построение графиков автокоррелограмм с точками и линиями уровня значимости
def plot_autocorrelogram(autocorrelations, p_values, critical_values, title):
    lags = range(1, len(autocorrelations) + 1)  # Lag начинается с 1
    significant_pos = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if corr > crit]
    significant_neg = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if corr < -crit]
    insignificant = [lag for lag, (corr, crit) in enumerate(zip(autocorrelations, critical_values), start=1) if -crit <= corr <= crit]

    plt.figure(figsize=(10, 6))
    plt.scatter(significant_pos, [autocorrelations[lag - 1] for lag in significant_pos], label='Положительные корреляции', s=10)
    plt.scatter(significant_neg, [autocorrelations[lag - 1] for lag in significant_neg], label='Отрицательные корреляции', s=10)
    plt.scatter(insignificant, [autocorrelations[lag - 1] for lag in insignificant], label='Незначимые корреляции', s=10)

    # Добавление линий уровня значимости

    #print(critical_values)
    negative_critical=[]
    for i in critical_values:
      negative_critical.append(i*-1)
    #plt.plot(critical_values,color='green')
    #plt.plot(negative_critical,color='orange')
    # plt.axhline(critical_values[0], color='green', linestyle='--', label=f'Верхний порог ({critical_values[0]:.3f})')
    # plt.axhline(-critical_values[0], color='orange', linestyle='--', label=f'Нижний порог ({-critical_values[0]:.3f})')
    plt.axhline(0, color='black', linewidth=0.5)

    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Корреляция')
    plt.legend()
    plt.show()

# Построение графиков
plot_autocorrelogram(ch_autocorr,ch_p, ch_critical, 'Чаттерджи')
plot_autocorrelogram(newch_autocorr,newch_p, newch_critical, 'Чаттерджи')
