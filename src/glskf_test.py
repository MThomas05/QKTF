import glskf
import numpy
import torch
import cupy as np
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter1d, gaussian_filter

def gen_synthetic_tensor(shape, rank, missing_fraction, target_local_std, df, seed, device):
    """
    Simple synthetic tensor generator with:
        Global structure: smooth low-rank structure.
        Local structure: spatial correlations.
        Heavy-tails: sparse extreme events.

    Args:
        shape (tuple): input tensor shape.
        rank (int): rank used in CP decomposition.
        missing_fraction (float): percentage of missing data entries.
        outlier_prob (float): probaility of an outlier occuring.
        seed (int): ensures reproducibility.
        device (string): CPU or GPU performance.

    Returns:
        ndarray: synthetic tensor to be used in QKTF algorithm.
    """
    torch.manual_seed(seed) # ensures reproducibility.
    numpy.random.seed(seed) # ensures reproducibility.

    D = len(shape) # gets the number of dimensions.

    # ========== Global structure ==========
    M_true = torch.zeros(shape, device=device) # initialises the true M as a zero tensor.
    for r in range(rank): # iterates over each rank component.
        factors = [] # stores each factor - used when generating one smooth factor per dimension.
        for d in range(D): # iterates over each dimension.
            u_d = numpy.random.randn(shape[d]) # random vector of length shape[d].
            u_d = gaussian_filter(u_d, sigma=5) # smooths the vector - creates global pattern.
            factors.append(u_d) # stores smoothed factor.

        # Compute D-dimensional outer product.
        component = factors[0] # starts with first factor - builds rank-one component.

        for d in range(1, D): # iteratively adds dimensions.
            component = component[..., None] # add new axis at end - broadcasts to next dim. '...' means all preceding dimensions.
            component = component * factors[d] # outer product with next factor.

        M_true += torch.tensor(component, dtype=torch.float32, device=device) # add this rank component to the true M.

    M_true = M_true / M_true.std() * 5 + 50 # normalise M to have a reasonable scale.

    # ========== Local structure ==========
    R_true = gaussian_filter(numpy.random.randn(*shape), sigma=0.5) # short lengthscale vs sigma for global.
    R_true = R_true / R_true.std() * target_local_std 
    R_true = torch.tensor(R_true, dtype=torch.float32, device=device)

    # ========== Heavy tails ==========
    noise = torch.distributions.StudentT(df=df, loc=0, scale=1) # generates tensor with Student-T distribution.
    dist = noise.sample(shape).to(device) # reshapes data to input tensor shape and sets to GPU performance.

    # ========== Tensor ==========
    tensor = M_true + R_true + dist # actual observed data.

    # ========== Mask creation ==========
    Omega = torch.rand(shape, device=device) >= missing_fraction # missing entries where random values are less than missing_fraction.

    return tensor, Omega, M_true, R_true, dist

def compute_diagnostics(tensor, Omega, X, M_true, R_true, M_pred, R_pred):
    """
    Compute comprehensive diagnostics for tensor completion.

    Args:
        tensor (ndaray): Original input tensor.
        X (ndarray): Estimated complete tensor - M + R.
        M_true (ndarray): True M (global) tensor.
        R_true (ndarray): True R (local) tensor.
        M_pred (ndarray): Predicted M (global) tensor.
        R_pred (ndarray): Predicted R (local) tensor.

    Returns:
    dict: Dictionary of diagnostic metrics.
    """

    obs_mask = Omega.astype(bool)
    miss_mask = ~obs_mask
    
    # ========== Reconstruction metrics ==========
    metrics = {}

    # Full tensor.
    error_full = tensor - X
    metrics['RMSE_full'] = np.sqrt(np.mean(error_full**2))
    metrics['Bias_full'] = np.mean(error_full)
    metrics['Variance_full'] = np.var(error_full)
    metrics['Std_full'] = np.std(error_full)
    metrics['Full_recovery'] = 1 - np.linalg.norm(error_full) / np.linalg.norm(tensor) # measures how well the full tensor is recovered.
    
    # ========== Component-wise metrics =========
    # Global component
    M_error = M_true - M_pred
    metrics['M_RMSE'] = np.sqrt(np.mean(M_error**2))
    metrics['M_Bias'] = np.mean(M_error)
    metrics['M_Variance'] = np.var(M_error)
    metrics['M_Std'] = np.std(M_error)
    metrics['M_recovery'] = 1 - np.linalg.norm(M_error) / np.linalg.norm(M_true) # measures how well M is recovered.

    # Local component
    R_error = R_true - R_pred
    metrics['R_RMSE'] = np.sqrt(np.mean(R_error**2))
    metrics['R_Bias'] = np.mean(R_error)
    metrics['R_Variance'] = np.var(R_error)
    metrics['R_Std'] = np.std(R_error)

    return metrics

def print_diagnostics(metrics):
    sections = {
        'Full': ['RMSE_full', 'Bias_full', 'Variance_full', 'Std_full', 'Full_recovery'],
        'Global M': ['M_RMSE', 'M_Bias', 'M_Variance', 'M_Std', 'M_recovery'],
        'Local R': ['R_RMSE', 'R_Bias', 'R_Variance', 'R_Std', 'R_recovery']
    }
    rows = []
    for section, keys in sections.items():
        for k in keys:
            label = k.split('_', 1)[-1] if '_' in k else k
            rows.append({'Section': section, 'Metric': label,
                         'Value': float(numpy.array(metrics[k].get()).ravel()[0])})
            
    df = (pd.DataFrame(rows)
          .set_index(['Section', 'Metric']))
    
    print(df.to_string(float_format='{:.6f}'.format))

seed = 1
device = 'cuda'
tensor_shape = (50, 50, 100, 100)
target_local_std = 2.0
df = 2
rank = 4
missing_fraction = 0.2
I, Omega, M_true, R_true, noise = gen_synthetic_tensor(tensor_shape, rank, missing_fraction, target_local_std, df, seed, device)
I = np.array(I)
Omega = np.array(Omega)

test_iter = 0

params_test = {
                'lengthscaleU': [5, 5, 5, 5],
                'lengthscaleR': [2, 2, 2, 2],
                'varianceU': [1, 1, 1, 1],
                'varianceR': [1, 1, 1, 1],
                'tapering_range': 4,
                'd_MaternU': 3,
                'd_MaternR': 3,
                'R': 3,
                'rho': 5, # Increase by 5 each test - until 20.
                'gamma': 5,
                'maxiter': 100,
                'K0': 10,
                'epsilon': 1e-8}
X, Rtensor, M = glskf.GLSKF(I, Omega, **params_test)

metrics = compute_diagnostics(I, Omega, X, M_true, R_true, M, Rtensor)
print_diagnostics(metrics)