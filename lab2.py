import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Загрузка данных ===
file_path = "EURUSD30.csv"
df = pd.read_csv(file_path, sep=",", header=None, names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"])

# Преобразуем дату и время в единое поле datetime
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M")

# Сортируем по времени
df = df.sort_values("Datetime").reset_index(drop=True)

# === 2. Выборка с интервалом T = 150 минут ===
T = 150
df_sampled = df.iloc[::T, :][["Datetime", "Open"]].dropna().reset_index(drop=True)

# === 3. Вычисление логарифмических доходностей ===
df_sampled["Log_Returns"] = np.log(df_sampled["Open"]).diff().dropna()

# Убираем NaN, которые появились после дифференцирования
df_sampled = df_sampled.dropna().reset_index(drop=True)

# === 4. Оценка параметров методом максимального правдоподобия (MLE) ===
rdist_params = stats.rdist.fit(df_sampled["Log_Returns"], floc=0)
foldcauchy_params = stats.foldcauchy.fit(df_sampled["Log_Returns"], floc=0)

print("Оценка параметров rdist (a, loc, scale):", rdist_params)
print("Оценка параметров foldcauchy (c, loc, scale):", foldcauchy_params)

# === 5. Определение числа интервалов по формуле Стёрджеса ===
num_bins = int(1 + np.log2(len(df_sampled)))

# === 6. Разбиение данных на интервалы с равными вероятностями ===
quantiles = np.linspace(0, 1, num_bins + 1)
bin_edges = np.quantile(df_sampled["Log_Returns"], quantiles)

# === 7. Группировка данных ===
bin_counts, _ = np.histogram(df_sampled["Log_Returns"], bins=bin_edges)

# === 8. Повторная оценка параметров MLE для группированных данных ===
rdist_grouped_params = stats.rdist.fit(bin_counts, floc=0)
foldcauchy_grouped_params = stats.foldcauchy.fit(bin_counts, floc=0)

print("Оценка параметров rdist по группированным данным:", rdist_grouped_params)
print("Оценка параметров foldcauchy по группированным данным:", foldcauchy_grouped_params)

# === 9. Вычисление статистики хи-квадрат ===
expected_counts_rdist = stats.rdist.cdf(bin_edges[1:], *rdist_params) - stats.rdist.cdf(bin_edges[:-1], *rdist_params)
expected_counts_rdist *= len(df_sampled)

expected_counts_foldcauchy = stats.foldcauchy.cdf(bin_edges[1:], *foldcauchy_params) - stats.foldcauchy.cdf(bin_edges[:-1], *foldcauchy_params)
expected_counts_foldcauchy *= len(df_sampled)

# === Исправление несоответствия сумм ===
expected_counts_rdist *= np.sum(bin_counts) / np.sum(expected_counts_rdist)
expected_counts_rdist = np.maximum(expected_counts_rdist, 1e-10)  # Убираем возможные нули

expected_counts_foldcauchy *= np.sum(bin_counts) / np.sum(expected_counts_foldcauchy)
expected_counts_foldcauchy = np.maximum(expected_counts_foldcauchy, 1e-10)

# === Расчёт критерия хи-квадрат ===
chi2_rdist, p_rdist = stats.chisquare(bin_counts, expected_counts_rdist)
chi2_foldcauchy, p_foldcauchy = stats.chisquare(bin_counts, expected_counts_foldcauchy)

print(f"Хи-квадрат для rdist: {chi2_rdist}, p-value: {p_rdist}")
print(f"Хи-квадрат для foldcauchy: {chi2_foldcauchy}, p-value: {p_foldcauchy}")

# === 10. Построение графиков ===
axes = plt.subplot(1, 1, 1 )

# Гистограмма
sns.histplot(df_sampled["Log_Returns"], bins=num_bins, kde=False, ax=axes)
axes.set_title("Гистограмма логарифмических доходностей")
axes.set_xlabel("Логарифмическая доходность")
axes.set_ylabel("Частота")

# Оценка плотности
x = np.linspace(df_sampled["Log_Returns"].min(), df_sampled["Log_Returns"].max(), 1000)
pdf_rdist = stats.rdist.pdf(x, *rdist_params)
pdf_foldcauchy = stats.foldcauchy.pdf(x, *foldcauchy_params)

axes.plot(x, pdf_rdist, label="rdist", color="blue")
axes.plot(x, pdf_foldcauchy, label="foldcauchy", color="red")
axes.set_title("Оценка плотности распределения")
axes.set_xlabel("Логарифмическая доходность")
axes.set_ylabel("Плотность вероятности")
axes.legend()

plt.tight_layout()
plt.show()