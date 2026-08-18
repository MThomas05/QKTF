import QKTFglobal
from data_gen import set_all_seeds
import cupy as np
import time

def run_QKTFglobal(I, Omega, signal, seed, tau):
    psi = 0.001
    sigma = 0.025
    rows = []

    set_all_seeds(seed)
    
    params = {
            'lengthscaleU': [4, 4, 4, 4],
            'varianceU': [1, 1, 1, 1],
            'd_MaternU': 3,
            'R': 6,
            'psi': psi,
            'sigma': sigma,
            'tau': tau,
            'max_iter': 200,
            'epsilon': 1e-4,
            'inner_maxiter': 500,
            'seed': seed}

    start = time.perf_counter()
    X, M = QKTFglobal.QKTFglobal(I, Omega, **params)
    np.cuda.Stream.null.synchronize()  # Ensure all GPU computations are complete.
    runtime = time.perf_counter() - start

    pinball = float(np.mean(np.where(signal[~Omega] - X[~Omega] >= 0,
                                     tau * (signal[~Omega] - X[~Omega]), (1 - tau) * -(signal[~Omega] - X[~Omega]))))
    medae = float(np.median(np.abs(signal[~Omega] - X[~Omega])))
    mae = float(np.mean(np.abs(signal[~Omega] - X[~Omega])))
    rmse = float(np.sqrt(np.mean((signal[~Omega] - X[~Omega]) ** 2)))
    recovery = float(1 - np.linalg.norm(signal[~Omega] - X[~Omega]) / np.linalg.norm(signal[~Omega]))
    error = float(np.linalg.norm(signal[~Omega] - X[~Omega]) / np.linalg.norm(signal[~Omega]))


    rows.append({'seed': seed, 'method': 'QKTFglobal', 'psi': psi, 'sigma': sigma, 'pinball': pinball, 'test_mae': mae, 'test_medae': medae,
                 'test_rmse': rmse, 'test_recovery': recovery, 'test_error': error, 'runtime': runtime})
    
    return rows

