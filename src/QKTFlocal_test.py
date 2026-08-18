import QKTFlocal
from data_gen import set_all_seeds
import cupy as np
import time

def run_QKTFlocal(I, Omega, signal, seed, tau):
    gamma = 0.5
    lambda_ = 0.015
    rows = []

    set_all_seeds(seed)
    
    params = { 
            'lengthscaleR': [1, 1, 1, 1],
            'varianceR': [1, 1, 1, 1],
            'tapering_range': 5,
            'd_MaternR': 3,
            'R': 6,
            'gamma': gamma, 
            'lambda_': lambda_,
            'tau': tau, 
            'max_iter': 200,
            'epsilon': 1e-4,
            'inner_maxiter': 500}

    start = time.perf_counter()
    X, Rtensor = QKTFlocal.QKTFlocal(I, Omega, **params)
    np.cuda.Stream.null.synchronize()  # Ensure all GPU computations are complete.
    runtime = time.perf_counter() - start

    print(f"QKTFlocal Rtensor (var): {float(np.var(Rtensor)):.4f}")

    pinball = float(np.mean(np.where(signal[~Omega] - X[~Omega] >= 0,
                                     tau * (signal[~Omega] - X[~Omega]), (1 - tau) * (X[~Omega] - signal[~Omega]))))
    medae = float(np.median(np.abs(signal[~Omega] - X[~Omega])))
    mae = float(np.mean(np.abs(signal[~Omega] - X[~Omega])))
    rmse = float(np.sqrt(np.mean((signal[~Omega] - X[~Omega]) ** 2)))
    recovery = float(1 - np.linalg.norm(signal[~Omega] - X[~Omega]) / np.linalg.norm(signal[~Omega]))
    error = float(np.linalg.norm(signal[~Omega] - X[~Omega]) / np.linalg.norm(signal[~Omega]))

    rows.append({'seed': seed, 'method': 'QKTFlocal', 'gamma': gamma, 'lambda_': lambda_, 'pinball': pinball, 'test_mae': mae, 'test_medae': medae,
                 'test_rmse': rmse, 'test_recovery': recovery, 'test_error': error, 'runtime': runtime})
    
    return rows
