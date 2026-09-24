import qktf
from data_gen import set_all_seeds
import cupy as np
import time

def run_qktf(I, Omega, signal, seed, tau):
    psi = 1e-2
    sigma = 1e-5
    gamma = 5
    lambda_ = 1e-2
    rows = []

    set_all_seeds(seed)
    
    params = {
        'lengthscaleU': [20, 20], 
        'lengthscaleR': [5, 5], 
        'varianceU': [1, 1],
        'varianceR': [1, 1],
        'tapering_range': 15, 
        'd_MaternU': 3,
        'd_MaternR': 3,
        'R': 15,
        'psi': psi,
        'sigma': sigma,
        'gamma': gamma, 
        'lambda_': lambda_,
        'tau': tau, 
        'max_iter': 100,
        'K0': 10,
        'epsilon': 1e-4,
        'inner_maxiter': 500,
        'seed': seed}

    np.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    X, Rtensor, M = qktf.qktf(I, Omega, **params)
    np.cuda.Stream.null.synchronize()  
    runtime = time.perf_counter() - start

    pinball = float(np.mean(np.where(signal[~Omega] - X[~Omega] >= 0,
                                     tau*(signal[~Omega] - X[~Omega]),
                                     (1 - tau)*-(signal[~Omega] - X[~Omega]))))
    medae = float(np.median(np.abs(signal[~Omega] - X[~Omega])))
    mae = float(np.mean(np.abs(signal[~Omega] - X[~Omega])))
    rmse = float(np.sqrt(np.mean((signal[~Omega] - X[~Omega])**2)))
    recovery = float(1 - np.linalg.norm(signal[~Omega] - X[~Omega]) 
                     / np.linalg.norm(signal[~Omega]))
    error = float(np.linalg.norm(signal[~Omega] - X[~Omega]) 
                  / np.linalg.norm(signal[~Omega]))

    rows.append({'seed': seed, 'method': 'QKTF', 
                 'psi': psi, 'sigma': sigma, 
                 'gamma': gamma, 'lambda_': lambda_, 'tau': tau, 
                 'pinball': pinball, 'test_mae': mae, 'test_medae': medae,
                 'test_rmse': rmse, 'test_recovery': recovery, 
                 'test_error': error, 'runtime': runtime})
    
    return rows
