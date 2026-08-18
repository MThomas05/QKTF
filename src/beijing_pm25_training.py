import numpy as np
import cupy as cp
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

# ----- Reproducibility -----
np.random.seed(seed)
cp.random.seed(seed)

# ----- Training hyper-parameters -----
eval_qktf = []
eval_glskf = []
psi = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
sigma = [1.0]
qktf_gamma = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
lambda_ = [1.0]
tau = [0.5]

rho = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
glskf_gamma = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]

qktf_params = {
    "lengthscaleU": [], "lengthscaleR": [],
    "varianceU": [], "varianceR": [],
    "d_MaternU": [], "d_MaternR": [],
    "tapering_range": , "R":  ,
    "psi": psi, "sigma": sigma, "gamma": qktf_gamma, "lambda_": lambda_, "tau": tau,
    "inner_maxiter": 500, "max_iter": 100, "K0": 10,
    "distance_matrix": d_station, "seed": seed, "epsilon": 1e-4
}

glskf_params = {
    "lengthscaleU": [], "lengthscaleR": [],
    "varianceU": [], "varianceR": [],
    "d_MaternU": [], "d_MaternR": [],
    "tapering_range": , "R": ,
    "rho": rho, "gamma": glskf_gamma,
    "max_iter": 100, "K0": 10,
    "distance_matrix": d_station, "seed": seed, "epsilon": 1e-4 
}

# ----- Run methods -----
for psi, sigma, qktf_gamma, lamdbda_ in itertools.product(psi, sigma, qktf_gamma, lambda_):
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    qktf_x, qktf_m, qktf_rtensor = qktf(I_train.copy(), mask_train.copy(), **qktf_params)
    runtime = time.perf_counter() - start
    start = time.perf_counter()

    # ----- Evaluation metrics -----
    qktf_pinball = float(cp.mean(cp.where(I_true[mask_val] - qktf_x[mask_val] >= 0,
                                        tau * (I_true[mask_val] - qktf_x[mask_val]), (1 - tau) * -(I_true[mask_val] - qktf_x[mask_val]))))
    qktf_test_medae = float(cp.median(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_test_mae = float(cp.mean(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_test_rmse = float(cp.sqrt(cp.mean((I_true[mask_val] - qktf_x[mask_val])**2)))
    qktf_test_recovery = float(1 - cp.linalg.norm((I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val])))
    qktf_test_error = float(cp.linalg.norm(I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val]))
    qktf_rtensor_norm = float(cp.linalg.norm(qktf_rtensor))
    qktf_m_norm = float(cp.linalg.norm(qktf_m))
    qktf_x_norm = float(cp.linalg.norm(qktf_x))
    
    iter += 1
    
    eval_qktf.append({
        'psi': psi,
        'sigma': sigma,
        'qktf_gamma': qktf_gamma,
        'lambda_': lambda_,
        'tau': tau,
        'rtensor_norm': qktf_rtensor_norm,
        'm_norm': qktf_m_norm,
        'x_norm': qktf_x_norm,
        'test_pinball': qktf_pinball,
        'test_rmse': qktf_test_rmse,
        'test_recovery': qktf_test_recovery,
        'test_error': qktf_test_error,
        'test_mae': qktf_test_mae,
        'test_medae': qktf_test_medae,
        'runtime': runtime})

for rho, glskf_gamma in itertools.product(rho, glskf_gamma):
    cp.cuda.Stream.null.Synchronize()
    start = time.perf_counter()
    glskf_x, glskf_m, glskf_rtensor = GLSKF(I_train.copy(), mask_train.copy(), **glskf_params)
    runtime = time.perf_counter() - start
    start = time.perf_counter()
    
    # ----- Evaluation metrics -----
    qktf_pinball = float(cp.mean(cp.where(I_true[mask_val] - qktf_x[mask_val] >= 0,
                                        tau * (I_true[mask_val] - qktf_x[mask_val]), (1 - tau) * -(I_true[mask_val] - qktf_x[mask_val]))))
    qktf_test_medae = float(cp.median(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_test_mae = float(cp.mean(cp.abs(I_true[mask_val] - qktf_x[mask_val])))
    qktf_test_rmse = float(cp.sqrt(cp.mean((I_true[mask_val] - qktf_x[mask_val])**2)))
    qktf_test_recovery = float(1 - cp.linalg.norm((I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val])))
    qktf_test_error = float(cp.linalg.norm(I_true[mask_val] - qktf_x[mask_val]) / cp.linalg.norm(I_true[mask_val]))
    qktf_rtensor_norm = float(cp.linalg.norm(qktf_rtensor))
    qktf_m_norm = float(cp.linalg.norm(qktf_m))
    qktf_x_norm = float(cp.linalg.norm(qktf_x))

    iter += 1
    
    eval_glskf.append({
        'psi': psi,
        'sigma': sigma,
        'qktf_gamma': qktf_gamma,
        'lambda_': lambda_,
        'tau': tau,
        'rtensor_norm': qktf_rtensor_norm,
        'm_norm': qktf_m_norm,
        'x_norm': qktf_x_norm,
        'test_pinball': qktf_pinball,
        'test_rmse': qktf_test_rmse,
        'test_recovery': qktf_test_recovery,
        'test_error': qktf_test_error,
        'test_mae': qktf_test_mae,
        'test_medae': qktf_test_medae,
        'runtime': runtime})
    


