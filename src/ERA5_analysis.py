from ERA5_preprocessing import load_ERA5, file_path
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import statistics

precip = load_ERA5(file_path)
precip_obs = precip.values
precip_obs = precip_obs[~np.isnan(precip_obs)]

print(np.isnan(precip_obs).sum())

# ----- Descriptive Statistics -----
quantiles = precip.quantile(
    [0.25, 0.5, 0.75, 0.90, 0.95, 0.99],
    skipna=True
)
q25 = quantiles.sel(quantile=0.25).item()
q50 = quantiles.sel(quantile=0.5).item()
q75 = quantiles.sel(quantile=0.75).item()
q90 = quantiles.sel(quantile=0.90).item()
q95 = quantiles.sel(quantile=0.95).item()
q99 = quantiles.sel(quantile=0.99).item()

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
        precip_obs.mean().item(),
        statistics.median(precip_obs),
        precip_obs.min().item(),
        precip_obs.max().item(),
        q25, q50, q75, q90, q95, q99,
        precip_obs.std().item(),
        q75 - q25,
        skew(precip_obs),
        kurtosis(precip_obs),
        (precip_obs == 0).mean().item(),
        np.isnan(precip.values).mean().item()
    ]
}

df = pd.DataFrame(data_dict)
df.to_csv('results/summary_stats_ERA5.csv')
print(df.to_string(index=False))

# ----- Overall Data Visualisations -----
fig, axes = plt.subplots(2, 2)
title_fs = 8
label_fs = 9
tick_fs = 8

# ----- Box plot -----
axes[0, 0].boxplot(precip_obs)
axes[0, 0].set_title("Box-plot of ERA5 precipitation observations", 
                     fontsize=title_fs)
axes[0, 0].set_ylabel("Precipitation (mm)", 
                      fontsize=label_fs)

# ---- Histogram -----
axes[0, 1].hist(precip_obs, bins=50)
axes[0, 1].axvline(precip_obs.mean(), color="r", 
                   linestyle="dashed", linewidth=2, 
                   label=f"Mean={precip_obs.mean():.2f}")
axes[0, 1].axvline(statistics.median(precip_obs), color="g",
                   linestyle="dotted", linewidth=2,
                   label=f"Median={statistics.median(precip_obs):.2f}")
axes[0, 1].set_title("Histogram of ERA5 precipitation observations",
                     fontsize=title_fs)
axes[0, 1].set_ylabel("Frequency", fontsize=label_fs)
axes[0, 1].set_xlabel("Precipitation (mm)", fontsize=label_fs)
axes[0, 1].legend

# ----- QQ-plot against Gaussian distribution -----
sm.qqplot(precip_obs, line="s", ax=axes[1, 0])
axes[1, 0].set_title("Normal Q-Q plot of ERA5 precipitation observations",
                     fontsize=title_fs)
axes[1, 0].set_ylabel("Theoretical Quantiles", fontsize=label_fs)
axes[1, 0].set_xlabel("Sample Quantiles", fontsize=label_fs)

# ----- Empirical Density -----
sns.kdeplot(precip_obs, ax=axes[1, 1])
axes[1, 1].set_title("Empirical Density Plot of ERA5 precipitation observations",
                     fontsize=title_fs)
axes[1, 1].set_xlabel("Precipitation (mm)", fontsize=label_fs)
axes[1, 1].set_ylabel("Density", fontsize=label_fs)

for ax in axes.flat:
    ax.tick_params(axis="both", labelsize=tick_fs)

plt.tight_layout()
plt.savefig("results/ERA5_precipitation_visualisations.png", dpi=300)
plt.show()