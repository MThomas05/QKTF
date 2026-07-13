import glskf
import cupy as np
from data_gen import set_all_seeds

def run_glskf(I, Omega, signal, test_mask, seed):
    rho = 10
    gamma = 20

    rows = []

    set_all_seeds(seed)

    params = {
        'rho': rho,
        'gamma': gamma,
        'lengthscaleU': [3, 3, 3, 3], # 10-20% of dimension size for global structure.
            'lengthscaleR': [1, 1, 1, 1], # 5-10% of dimension size for local structure.
            'varianceU': [1, 1, 1, 1],
            'varianceR': [1, 1, 1, 1],
            'tapering_range': 2, # At least x2 the local lengthscale to capture local structure.
            'd_MaternU': 3,
            'd_MaternR': 3,
            'R': 3,
            'max_iter': 200,
            'K0': 5,
            'epsilon': 1e-5,
            'seed': seed}

    X, Rtensor, M = glskf.GLSKF(I, Omega, **params)

    rmse = float(np.sqrt(np.mean((signal[test_mask] - X[test_mask])**2)))
    recovery = float(1 - np.linalg.norm((signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask])))
    error = float(np.linalg.norm(signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask]))

    rows.append({'seed': seed, 'method': 'GLSKF', 'rho': rho, 'gamma': gamma,
                 'test_rmse': rmse, 'test_recovery': recovery, 'test_error': error})
    
    return rows