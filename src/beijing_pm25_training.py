import numpy as np
import cupy as cp
import pandas as pd
from qktf import qktf
from glskf import GLSKF
import itertools
import time

# ----- Data loading -----
seed = 123
train = np.load("Data/splits/beijing_pm25_training.npz")
I_train = cp.asarray(train["I_train"])
mask_train = cp.asarray(train["mask_train"])
d_station = cp.asarray(train["d_station"])
I_true = cp.asarray(train["I_true"])
I_val = cp.asarray(train["I_val"])
mask_val = cp.asarray(train["mask_val"])

distance_matrices = [d_station, None, None] # representing: station, day, hour

# ----- Correctness checking -----
assert cp.all(I_train[mask_val] == 0)
assert cp.array_equal(I_val[mask_val], I_true[mask_val])
assert not cp.any(mask_train & mask_val)

# ----- Hyper-parameter values -----
eval_qktf = []
eval_glskf = []

psi = [5.0]
sigma = [0.05, 0.1, 0.5, 1, 5, 10]
qktf_gamma = [10.0]
lambda_ = [0.05, 0.1, 0.5, 1, 5, 10]
tau = 0.5

rho = [5, 10, 15]
glskf_gamma = [10, 20, 30]

# ----- QKTF training -----
qktf_iter = 0
glskf_iter = 0

for psi, sigma, qktf_gamma, lambda_ in itertools.product(psi, sigma, qktf_gamma, lambda_):
    cp.random.seed(seed)
    np.random.seed(seed)

    qktf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 15, "R": 8,
    "psi": psi, "sigma": sigma, "gamma": qktf_gamma, "lambda_": lambda_, "tau": 0.5,
    "inner_maxiter": 500, "max_iter": 100, "K0": 10,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4
     }
    
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktf_x, qktf_rtensor, qktf_m = qktf(I_train.copy(), mask_train.copy(), **qktf_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    # ----- Evaluation metrics -----
    qktf_val_pinball = float(cp.mean(cp.where(I_true[mask_val] - qktf_x[mask_val] >= 0,
                                        tau * (I_true[mask_val] - qktf_x[mask_val]), (1 - tau) * -(I_true[mask_val] - qktf_x[mask_val]))))
    qktf_val_medae = float(cp.median(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_val_mae = float(cp.mean(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_val_rmse = float(cp.sqrt(cp.mean((I_true[mask_val] - qktf_x[mask_val])**2)))
    qktf_val_recovery = float(1 - cp.linalg.norm((I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val])))
    qktf_val_error = float(cp.linalg.norm(I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val]))
    qktf_rtensor_norm = float(cp.linalg.norm(qktf_rtensor))
    qktf_m_norm = float(cp.linalg.norm(qktf_m))
    qktf_x_norm = float(cp.linalg.norm(qktf_x))
    
    qktf_iter += 1
    
    eval_qktf.append({
        'psi': psi,
        'sigma': sigma,
        'qktf_gamma': qktf_gamma,
        'lambda_': lambda_,
        'tau': tau,
        'rtensor_norm': qktf_rtensor_norm,
        'm_norm': qktf_m_norm,
        'x_norm': qktf_x_norm,
        'val_pinball': qktf_val_pinball,
        'val_rmse': qktf_val_rmse,
        'val_recovery': qktf_val_recovery,
        'val_error': qktf_val_error,
        'val_mae': qktf_val_mae,
        'val_medae': qktf_val_medae,
        'runtime': runtime})

eval_qktf_df = pd.DataFrame(eval_qktf)
eval_qktf_df.to_csv("results/qktf_hyper-parameter_training_beijing_pm25.csv")
print(eval_qktf_df.to_string())

# ----- GLSKF training -----
for rho, glskf_gamma in itertools.product(rho, glskf_gamma):
    cp.random.seed(seed)
    np.random.seed(seed)

    glskf_params = {
    "lengthscaleU": [30.0, 8.0], "lengthscaleR": [7.5, 2.0],
    "varianceU": [1.0, 1.0], "varianceR": [1.0, 1.0],
    "d_MaternU": 3, "d_MaternR": 3,
    "tapering_range": 20, "R": 8,
    "rho": rho, "gamma": glskf_gamma,
    "maxiter": 100, "K0": 10,
    "distance_matrix": distance_matrices, "seed": seed, "epsilon": 1e-4 
    }

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    glskf_x, glskf_rtensor, glskf_m = GLSKF(I_train.copy(), mask_train.copy(), **glskf_params)
    cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - start

    # ----- Evaluation metrics -----
    glskf_val_pinball = float(cp.mean(cp.where(I_true[mask_val] - glskf_x[mask_val] >= 0,
                                        tau * (I_true[mask_val] - glskf_x[mask_val]), (1 - tau) * -(I_true[mask_val] - glskf_x[mask_val]))))
    glskf_val_medae = float(cp.median(cp.abs(I_true[mask_val] - glskf_x[mask_val])))
    glskf_val_mae = float(cp.mean(cp.abs(I_true[mask_val] - glskf_x[mask_val])))
    glskf_val_rmse = float(cp.sqrt(cp.mean((I_true[mask_val] - glskf_x[mask_val])**2)))
    glskf_val_recovery = float(1 - cp.linalg.norm((I_true[mask_val] - glskf_x[mask_val]) / cp.linalg.norm(I_true[mask_val])))
    glskf_val_error = float(cp.linalg.norm(I_true[mask_val] - glskf_x[mask_val]) / cp.linalg.norm(I_true[mask_val]))
    glskf_rtensor_norm = float(cp.linalg.norm(glskf_rtensor))
    glskf_m_norm = float(cp.linalg.norm(glskf_m))
    glskf_x_norm = float(cp.linalg.norm(glskf_x))

    glskf_iter += 1
    
    eval_glskf.append({
        'rho': rho,
        'glskf_gamma': glskf_gamma,
        'rtensor_norm': glskf_rtensor_norm,
        'm_norm': glskf_m_norm,
        'x_norm': glskf_x_norm,
        'val_pinball': glskf_val_pinball,
        'val_rmse': glskf_val_rmse,
        'val_recovery': glskf_val_recovery,
        'val_error': glskf_val_error,
        'val_mae': glskf_val_mae,
        'val_medae': glskf_val_medae,
        'runtime': runtime})

eval_glskf_df = pd.DataFrame(eval_glskf)
eval_glskf_df.to_csv("results/glskf_hyper-parameter_training_beijing_pm25.csv")
print(eval_glskf_df.to_string())
    


