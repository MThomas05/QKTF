from beijing_pm25_preprocessing import load_beijing_pm25, haversine, station_dist, folder
from qktf import qktf
from QKTFlocal import QKTFlocal
from QKTFglobal import QKTFglobal
from glskf import glskf
import pandas as pd
import numpy as np
import cupy as cp
import time

# ----- Reproducibility -----
seed = 1234
cp.random.seed(seed)
np.random.seed(seed)

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

distance_matrices = [d_station, None, None]

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
def evaluate_method(method, X, Rtensor, M, I, mask_train, mask_test, missing, runtime):
    """
    Evaluate reconstructoin on artificially hidden observations only.
    
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
    rmse = float(cp.sqrt(cp.mean(I[mask_test] - X[mask_test]) ** 2))
    recovery = float(1 - cp.linalg.norm(I[mask_test] - X[mask_test]) / cp.linalg.norm(I[mask_test] - X[mask_test]))

    metrics = {
        "method": method, "missing": missing,
        "n_test": int(mask_train.sum()), "n_train": int(mask_test.sum()),
        "test_mae": mae, "test_medae": medae, "test_rmse": rmse, "test_rr": recovery,
        "rtensor_norm": float(cp.linalg.norm(Rtensor)), "m_norm": float(cp.linalg.norm(M)),
        "x_norm": float(cp.linalg.norm(X)), "runtime": runtime
    }

    return metrics

# ----- Hyper-parameter setting -----
sigma, lambda_, qktf_gamma, psi, tau = 1e-3, 1e-3, 1e-4, 10, 0.5
rho, glskf_gamma = 15, 30
qktf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "psi": psi, "sigma": sigma, "gamma": qktf_gamma, "lambda_": lambda_, "tau": tau,
    "inner_maxiter": 500, "max_iter": 200, "K0": 40,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
}
qktflocal_params = {
    "lengthscaleU": [30.0, 8.0], "varianceR": [1.0, 1.0], "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "gamma": qktf_gamma, "lambda_": lambda_, "tau": tau,
    "inner_maxiter": 500, "max_iter": 200, "K0": 40,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
}
qktfglobal_params = {
    "lengthscaleU": [30.0, 8.0], "varianceU": [1.0, 1.0], "d_MaternU": 3,
    "tapering_range": 15, "R": 15,
    "psi": psi, "sigma": sigma, "tau": tau,
    "inner_maxiter": 500, "max_iter": 200, "K0": 40,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
}
glskf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 15, "R": 15,
    "rho": rho, "gamma": glskf_gamma,
    "maxiter": 200, "K0": 40,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4 
}

# ----- Experiment -----
missingness = [0.3, 0.5, 0.7]
all_rows = []

for missing in missingness:
    print(f"Missingness: {int(missingness*100)}%")

    # ----- Tensor construction -----
    I_true, I_train, I_test, mask_train, mask_test = mask_construct(I, mask_original, seed, missing)

    I_train = cp.asarray(I_train)
    I_test = cp.asarray(I_test)
    mask_train = cp.asarray(mask_train)
    mask_test = cp.asarray(mask_test)

    # ----- QKTF -----
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktf_x, qktf_rtensor, qktf_m = qktf(I_train.copy(), mask_train.copy(), **qktf_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    all_rows.append(evaluate_method(
        "QKTF", qktf_x, qktf_rtensor, qktf_m, I_true, mask_train, mask_test, missing, runtime
    ))

    # ----- QKTFlocal -----