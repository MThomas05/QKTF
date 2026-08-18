from beijing_pm25_preprocessing import load_beijing_pm25, station_dist, folder
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

pm25, station_coords = load_beijing_pm25(folder)
d_station = station_dist(pm25, station_coords)

pm25_obs = pm25.dropna()

# ----- Descriptive Statistics -----
pm25_values = pm25_obs["PM2.5"]

sum_stat = pm25_values.describe(
    percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
)

median_pm25 = pm25_values.median()
skew_pm25 = skew(pm25_values)
kurtosis_pm25 = kurtosis(pm25_values)

data_dict = {
    "Statistic": [
        "Mean", "Median", "Min", "Max",
        "25pp", "50pp", "75pp",
        "90pp", "95pp", "99pp",
        "Std", "IQR",
        "Skewness", "Excess kurtosis",
        "Zero proportion",
        "Missing proportion"
    ],
    "Stat_value": [
        sum_stat.loc["mean"],
        median_pm25,
        sum_stat.loc["min"],
        sum_stat.loc["max"],
        sum_stat.loc["25%"],
        sum_stat.loc["50%"],
        sum_stat.loc["75%"],
        sum_stat.loc["90%"],
        sum_stat.loc["95%"],
        sum_stat.loc["99%"],
        sum_stat.loc["std"],
        sum_stat.loc["75%"] - sum_stat.loc["25%"],
        skew_pm25,
        kurtosis_pm25,
        (pm25_values == 0).mean(),
        pm25["PM2.5"].isna().mean()
    ]
}
df = pd.DataFrame(data_dict)
df.to_csv('results/summary_stats_beijingpm25.csv')
print(df.to_string())

# ----- Overall Data Visualisations -----
fig, axes = plt.subplots(2, 2)

# ----- Box-plot -----
axes[0, 0].boxplot(pm25_values)
axes[0, 0].set_title("Box-plot of PM2.5 observations")
axes[0, 0].set_ylabel("PM2.5")

# ----- Histogram -----
axes[0, 1].hist(pm25_values, bins=50)
axes[0, 1].axvline(sum_stat.loc["mean"], color="red", linestyle="dashed", linewidth=2, label=f"Mean={sum_stat.loc['mean']:.2f}")
axes[0, 1].axvline(median_pm25, color="green", linestyle="dotted", linewidth=2, label=f"Median={median_pm25:.2f}")
axes[0, 1].set_title("Histogram of PM2.5 observations")
axes[0, 1].set_ylabel("PM2.5")
axes[0, 1].legend

# ----- QQ-plot against Gaussian distribution -----
sm.qqplot(pm25_values, line='s', ax=axes[1, 0])
axes[1, 0].set_title("Normal Q-Q plot of Pm2.5 observations")

# ----- Empirical Density -----
sns.kdeplot(pm25_values, ax=axes[1, 1])
axes[1, 1].set_title("Empirical Density Plot of PM2.5 observations")

plt.tight_layout()
plt.savefig("results/beijing_pm25_diststats.png", dpi=300, bbox_inches="tight")
plt.show()

# ----- Station Visualisations -----
plt.figure(figsize=(12, 6))
sns.boxplot(data=pm25_obs, x="station", y="PM2.5")
plt.xticks(rotation=45)
plt.title("Box-plot of PM2.5 observations by station")
plt.xlabel("Station")
plt.ylabel("PM2.5")
plt.tight_layout
plt.savefig("results/beijing_pm25_stationdist.png", dpi=300, bbox_inches="tight")
plt.show()

# ----- Temporal Structure -----
hourly = pm25_obs.groupby("hour")["PM2.5"].median()
daily = pm25_obs.groupby("date")["PM2.5"].median()

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(hourly.index, hourly.values)
ax[0].set_label("Hour")
ax[0].set_ylabel("Median PM2.5")
ax[0].set_title("Median PM2.5 by hour of day")

ax[1].plot(daily.index, daily.values)
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Median PM2.5")
ax[1].set_title("Daily median PM2.5")

plt.tight_layout
plt.savefig("results/beijing_pm25_temporalstruct.png", dpi=300, bbox_inches="tight")
plt.show()
