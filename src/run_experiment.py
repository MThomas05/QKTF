import pandas as pd
import config as cfg
import cupy as cp
from data_gen import get_or_create_tensor
from qktf_test import run_qktf
from glskf_test import run_glskf
from QKTFglobal_test import run_QKTFglobal
from QKTFlocal_test import run_QKTFlocal
import os

os.makedirs("results", exist_ok=True) # ensure results folder exists.

all_rows = []
tau = 0.75

for seed in cfg.SEEDS:
    # Same cached tensor per seed, so every method tests on identical data.
    I, Omega, M_true, R_true, noise = get_or_create_tensor(seed, cfg)

    signal = M_true + R_true

    # Run all four method on this seed's data before moving to next seed.
    all_rows.extend(run_glskf(I.copy(), Omega.copy(), signal, seed))
    all_rows.extend(run_qktf(I.copy(), Omega.copy(), signal, seed, tau=tau))
    all_rows.extend(run_QKTFglobal(I.copy(), Omega.copy(), signal, seed, tau=tau))
    all_rows.extend(run_QKTFlocal(I.copy(), Omega.copy(), signal, seed, tau=tau))

df = pd.DataFrame(all_rows)
df.to_csv("results/raw_results_LogNormal_(20,400,30)_90%_missing_tau=0.75.csv", index=False) # unaggregated results.

# ========== manual sanity check ==========
metric_cols = ['pinball','test_mae', 'test_medae', 'test_rmse', 'test_recovery', 'test_error', 'runtime']

for seed in cfg.SEEDS:
    print(f"\n=== Seed {seed} ===")
    seed_df = df[df['seed'] == seed][['method'] + metric_cols]
    print(seed_df.to_string(index=False)) # allows for manual calculation for sanity check.
                 
# ========== Aggregate across 5 seeds ==========
summary = (df.groupby('method')[metric_cols].agg(['mean', 'std'])) # one row per method.
summary.columns = ['_'.join(col) for col in summary.columns] # flattens multi-index columns
summary = summary.reset_index()

def fmt_mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

paper_table = summary[['method']].copy()
for metric in metric_cols:
    paper_table[metric] = [
        fmt_mean_std(m, s) for m, s in zip(summary[f'{metric}_mean'], summary[f'{metric}_std'])
    ]

print("\n=== Aggregated (mean ± std) across seeds ===")
print(paper_table.to_string(index=False)) # final numbers to copy into the results table.

paper_table.to_csv("results/paper_table_LogNormal_(20,400,30)_90%_missing_tau=0.75.csv", index=False)