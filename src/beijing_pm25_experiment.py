from beijing_pm25_preprocessing import load_beijing_pm25, haversine, station_dist, folder
from qktf import qktf
from QKTFlocal import QKTFlocal
from QKTFglobal import QKTFglobal
from glskf import GLSKF
import pandas as pd
import numpy as np
import cupy as cp
import time
import os

# ----- Reproducibility -----
seed = 1234

def reset_seed(seed):
    """
    Function that sets the seed.
    
    Inputs:
        (seed) int: seed for reproducibility"""
    cp.random.seed(seed)
    np.random.seed(seed)

reset_seed(seed)

os.makedirs("results", exist_ok=True)

pm25, station_coords = load_beijing_pm25(folder)
d_station = station_dist(pm25, station_coords)

# ----- Tensor construction (station x day x hour) -----
stations = sorted(pm25["station"].unique())
coords = np.array([station_coords[station] for station in stations], dtype=np.float64)
hours = list(range(24))
days = sorted(pm25["day"].unique())

index = pd.MultiIndex.from_product([stations, days, hours], names=["stations", "days", "hours"])
pm25_index = (pm25.set_index(["station", "day", "hour"])["PM2.5"].reindex(index))

I_numpy = pm25_index.to_numpy().reshape(len(stations), len(days), len(hours))
I = np.array(I_numpy) # complete input tensor

mask_original = ~np.isnan(I) # corresponds to the geninely observed tensor

distance_matrices = [cp.asarray(d_station), None, None]

assert I.shape == (12, 365, 24)
assert len(stations) == 12
assert len(days) == 365

# ----- Mask Construction -----
def mask_construct(I, mask_original, seed, missing):
    """
    Function that constructs the artifical mask
    
    Inputs:
        I (ndarray): input tensor
        mask_original (ndarray): original mask on the data - masks NaN values
        seed (int): for reproducibility
        missing (float): missingness fraction
    Outputs:
        mask (ndarray): artificial mask used for training and testing across all algorithms
        I (ndarray): input tensor"""
    rng = np.random.RandomState(seed)

    # Observed entries only
    observed_entries = np.where(mask_original.ravel())[0]
    observed_entries = rng.permutation(observed_entries)

    n_observed = len(observed_entries)
    n_missing = int(round(missing * n_observed))

    train_entries = observed_entries[n_missing:]
    test_entries = observed_entries[:n_missing]

    # ---- Mask construction -----
    mask_train = np.zeros(I.size, dtype=bool)
    mask_test = np.zeros(I.size, dtype=bool)
    mask_train[train_entries] = True
    mask_test[test_entries] = True
    mask_train = mask_train.reshape(I.shape)
    mask_test = mask_test.reshape(I.shape)

    assert not np.any(mask_train & mask_test)
    assert int(mask_train.sum() + mask_test.sum()) == int(mask_original.sum())

    # ----- True tensor -----
    I_true = np.where(mask_original, I, 0.0)

    # ----- Development tensor -----
    I_train = I_true.copy()
    I_train[~mask_train] = 0.0

    # ----- Testing tensor -----
    I_test = I_true.copy()
    I_test[~mask_test] = 0.0

    assert np.all(I_train[mask_test] == 0)
    assert np.all(I_test[mask_train] == 0)

    return I_true, I_train, I_test, mask_train, mask_test

# ----- Evaluation -----
def evaluate_method(method, X, I, 
                    mask_train, mask_test, missing, 
                    runtime, tau, 
                    M=None, Rtensor=None):
    """
    Evaluate reconstruction on artificially hidden observations only.
    
    Inputs:
        method (string): method that is run
        X (ndarray): output tensor from the method
        Rtensor (ndarray): output local tensor from the method
        M (ndarray): output global tensor from the method
        I (ndarray): input tensor
        mask (ndarray): mask placed on the input tensor
        missing (float): missingness fraction for the mask
        runtime (float): total runtime for each method"""

    # ----- Evaluation metrics -----
    mae = float(cp.mean(cp.abs(I[mask_test] - X[mask_test])))
    medae = float(cp.median(cp.abs(I[mask_test] - X[mask_test])))
    rmse = float(cp.sqrt(cp.mean((I[mask_test] - X[mask_test]) ** 2)))
    recovery = float(1 - (cp.linalg.norm(I[mask_test] - X[mask_test]) / cp.linalg.norm(I[mask_test])))
    pinball = float(cp.mean(cp.where(I[mask_test] - X[mask_test] >= 0,
                                         tau * (I[mask_test] - X[mask_test]), (1 - tau) * - (I[mask_test] - X[mask_test]))))

    metrics = {
        "method": method, "missing": missing,
        "n_train": int(mask_train.sum().item()), "n_test": int(mask_test.sum().item()),
        "test_pinball": pinball, "test_mae": mae, "test_medae": medae, "test_rmse": rmse, "test_rr": recovery,
        "rtensor_norm": float(cp.linalg.norm(Rtensor)) if Rtensor is not None else 0.0, 
        "m_norm": float(cp.linalg.norm(M)) if M is not None else 0.0,
        "x_norm": float(cp.linalg.norm(X)), "runtime": runtime
    }

    return metrics

# ----- Hyper-parameter setting -----
psi, sigma, qktf_gamma, lambda_, tau = 1e-3, 1e-4, 10, 1e-3, 0.5
rho, glskf_gamma = 15, 30
qktf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "psi": psi, "sigma": sigma, "gamma": qktf_gamma, "lambda_": lambda_, "tau": tau,
    "inner_maxiter": 500, "max_iter": 100, "K0": 10,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
}
qktflocal_params = {
    "lengthscaleR": [7.5, 2.0], "varianceR": [1.0, 1.0], "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "gamma": qktf_gamma, "lambda_": lambda_, "tau": tau,
    "inner_maxiter": 500, "max_iter": 100,
    "distance_matrix": distance_matrices, "epsilon": 1e-4
}
qktfglobal_params = {
    "lengthscaleU": [30.0, 8.0], "varianceU": [1.0, 1.0], "d_MaternU": 3,
    "R": 15, "psi": psi, "sigma": sigma, "tau": tau,
    "inner_maxiter": 500, "max_iter": 100,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
}
glskf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "rho": rho, "gamma": glskf_gamma,
    "maxiter": 100, "K0": 10,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4 
}

# ----- Experiment -----
missingness = [0.3, 0.5, 0.7]
all_rows = []

for missing in missingness:
    print(f"Missingness: {int(missing*100)}%")

    # ----- Tensor construction -----
    I_true, I_train, I_test, mask_train, mask_test = mask_construct(I, mask_original, seed, missing)

    np.savez(f"results/beijing_pm25_experiment_{int(100*missing)}.npz",
             I_true=I_true, I_train=I_train, mask_train=mask_train, mask_test=mask_test)

    I_train = cp.asarray(I_train)
    I_true = cp.asarray(I_true)
    mask_train = cp.asarray(mask_train)
    mask_test = cp.asarray(mask_test)

    assert int(mask_test.sum().item()) == int(round(missing * mask_original.sum()))

    # ----- QKTF -----
    reset_seed(seed)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktf_x, qktf_rtensor, qktf_m = qktf(I_train.copy(), mask_train.copy(), **qktf_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    all_rows.append(evaluate_method(
        "QKTF", qktf_x, I_true, mask_train, mask_test, missing, runtime, tau,
        M=qktf_m, Rtensor=qktf_rtensor
    ))

    # ----- QKTFlocal -----
    reset_seed(seed)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktflocal_x, qktflocal_rtensor = QKTFlocal(I_train.copy(), mask_train.copy(), **qktflocal_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    all_rows.append(evaluate_method(
        "QKTFlocal", qktflocal_x, I_true, mask_train, mask_test, missing, runtime, tau, 
        M=None, Rtensor=qktflocal_rtensor
    ))

    # ----- QKTFglobal -----
    reset_seed(seed)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktfglobal_x, qktfglobal_m = QKTFglobal(I_train.copy(), mask_train.copy(), **qktfglobal_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    all_rows.append(evaluate_method(
        "QKTFglobal", qktfglobal_x, I_true, mask_train, mask_test, missing, runtime, tau,
        M=qktfglobal_m, Rtensor=None
    ))

    # ----- GLSKF -----
    reset_seed(seed)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    glskf_x, glskf_rtensor, glskf_m = GLSKF(I_train.copy(), mask_train.copy(), **glskf_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    all_rows.append(evaluate_method(
        "GLSKF", glskf_x, I_true, mask_train, mask_test, missing, runtime, tau,
        M=glskf_m, Rtensor=glskf_rtensor
    ))

# ----- Results -----
results_df = pd.DataFrame(all_rows)
results_df.to_csv("results/final_beijing_pm25_results.csv", index=False)
print("\nFinal results:")
print(results_df.to_string(index=False))