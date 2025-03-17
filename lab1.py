import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest

data = pd.read_csv('EURUSD30.csv', header=None)
data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('DateTime', inplace=True)
open_prices = data['Open'].resample('150T').first().dropna()

price_changes = open_prices.diff().dropna()
print(price_changes)

t_stat_price, p_value_price = stats.ttest_1samp(price_changes, 0)
shapiro_stat_price, shapiro_p_value_price = shapiro(price_changes)
dagostino_stat_price, dagostino_p_value_price = normaltest(price_changes)

log_returns = np.log(open_prices / open_prices.shift(1)).dropna()

t_stat_log, p_value_log = stats.ttest_1samp(log_returns, 0)
shapiro_stat_log, shapiro_p_value_log = shapiro(log_returns)
dagostino_stat_log, dagostino_p_value_log = normaltest(log_returns)

print("Вычисление изменения цен:")
print(f"Объем выборки: {len(price_changes)}")
print(f"Тест Стьюдента: {t_stat_price}; p-значение: {p_value_price}")
print(f"ТестД Агостино и Пирсона: статистика -- {dagostino_stat_price}; p-значение -- {dagostino_p_value_price}")
print(f"Тест Шапиро-Уилка: статистика -- {shapiro_stat_price}; p-значение -- {shapiro_p_value_price}")

print("Вычисление логарифмических доходностей:")
print(f"Объем выборки: {len(log_returns)}")
print(f"Тест Стьюдента: {t_stat_log}; p-значение: {p_value_log}")
print(f"Тест Д Агостино и Пирсона: статистика -- {dagostino_stat_log}; p-значение -- {dagostino_p_value_log}")
print(f"Тест Шапиро-Уилка: статистика -- {shapiro_stat_log}; p-значение -- {shapiro_p_value_log}")