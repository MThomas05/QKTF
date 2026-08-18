import pandas as pd
import config as cfg
import cupy as cp
from data_gen import get_or_create_tensor
from qktf_test import run_qktf
from glskf_test import run_glskf
from QKTFglobal_test import run_QKTFglobal
from QKTFlocal_test import run_QKTFlocal
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True) # ensure results folder exists.

all_rows = []
tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def report(rows):
    for r in rows:
        tau = r.get('tau')
        tau_str = f"tau={float(tau):.1f}" if tau is not None else "tau= -"
        print(f"seed={r.get('seed', '?')} {tau_str} {r['method']:<11}"
              f"MAE={float(r['test_mae']):8.4f}"
              f"MedAE={float(r['test_medae']):8.4f}"
              f"RMSE={float(r['test_rmse']):8.4f}",
              flush=True)
    return rows

for seed in cfg.SEEDS:
    # Same cached tensor per seed, so every method tests on identical data.
    I, Omega, M_true, R_true, noise = get_or_create_tensor(seed, cfg)
    signal = M_true + R_true
    print(f"M_true_norm={cp.linalg.norm(M_true)}",
          f"R_true_norm={cp.linalg.norm(R_true)}",
          f"signal_norm={cp.linalg.norm(signal)}")
    print(f"|noise| median={float(cp.median(abs(noise))):.2f} "
          f"p99={float(cp.percentile(abs(noise),99)):.1f} max={float(abs(noise).max()):.1f}")

    # Run all four method on this seed's data before moving to next seed.

    for tau in tau_values:
        all_rows.extend(report(run_qktf(I.copy(), Omega.copy(), signal.copy(), seed, tau=tau)))

    all_rows.extend(report(run_glskf(I.copy(), Omega.copy(), signal.copy(), seed)))

df = pd.DataFrame(all_rows)
df.to_csv("results/raw_results_tau_sweep_Cauchy_(10,_10,_10,_10)_30% missing.csv", index=False) # unaggregated results.

# ========== manual sanity check ==========
metric_cols = ['test_mae', 'test_medae', 'test_rmse', 'test_recovery', 'test_error', 'runtime']

tau_dependent = df[df['method'] != 'GLSKF']
glskf_only = df[df['method'] == 'GLSKF']

summary_tau = tau_dependent.groupby(['method', 'tau'])[metric_cols].agg(['mean', 'std'])
summary_tau.columns = ['_'.join(c) for c in summary_tau.columns]
summary_tau = summary_tau.reset_index()

summary_glskf = glskf_only.groupby('method')[metric_cols].agg(['mean', 'std'])
summary_glskf.columns = ['_'.join(c) for c in summary_glskf.columns]
summary_glskf = summary_glskf.reset_index()

def fmt_mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

print("\n=== Tau sweep results (mean ± std across seeds) ===")
for tau in tau_values:
    print(f"\n--- tau = {tau} ---")
    block = summary_tau[summary_tau['tau'] == tau]
    table = block[['method']].copy()
    for metric in metric_cols:
        table[metric] = [fmt_mean_std(m, s) for m, s in zip(block[f'{metric}_mean'], block[f'{metric}_std'])]
    print(table.to_string(index=False))

print("\n--- GLSKF (tau-independent baseline) ---")
glskf_table = summary_glskf[['method']].copy()
for metric in metric_cols:
    glskf_table[metric] = [fmt_mean_std(m, s) for m, s in zip(summary_glskf[f'{metric}_mean'], summary_glskf[f'{metric}_std'])]
print(glskf_table.to_string(index=False))

summary_tau.to_csv("results/tau_sweep_summary_Cauchy_(10,_10,_10,_10)_30%_missing.csv", index=False)

# ========== Plots ==========
plot_specs = [('test_mae', 'MAE'), ('test_medae', 'MedAE')]
methods = ['QKTF']
colours = {'QKTF': 'tab:blue', 'GLSKF': 'tab:red'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, (metric_col, metric_label) in zip(axes, plot_specs):
    for method in methods:
        m_df = summary_tau[summary_tau['method'] == method].sort_values('tau')
        ax.errorbar(m_df['tau'], m_df[f'{metric_col}_mean'], yerr=m_df[f'{metric_col}_std'],
                    label=method, color=colours[method], marker='o', capsize=3)

    glskf_mean = summary_glskf[f'{metric_col}_mean'].iloc[0]
    glskf_std = summary_glskf[f'{metric_col}_std'].iloc[0]
    ax.axhline(glskf_mean, color=colours['GLSKF'], linestyle='--', label='GLSKF')
    ax.axhspan(glskf_mean - glskf_std, glskf_mean + glskf_std, color=colours['GLSKF'], alpha=0.1)

    ax.set_xlabel(r'Quantile level $\tau$')
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend()
    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

fig.tight_layout()
fig.savefig(f"results/Cauchy_(10,_10,_10,_10)_30%_missing_tau_sweep.png", dpi=200)
plt.show()