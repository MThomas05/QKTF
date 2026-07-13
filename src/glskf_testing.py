import glskf
import numpy
import torch
import cupy as np
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter1d, gaussian_filter

seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

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
    R_true = gaussian_filter(numpy.random.randn(*shape), sigma=2) # short lengthscale vs sigma for global.
    R_true = R_true / R_true.std() * target_local_std 
    R_true = torch.tensor(R_true, dtype=torch.float32, device=device)

    # ========== Heavy tails ==========
    noise = torch.distributions.StudentT(df=df, loc=0.0, scale=1.0) # generates tensor with Student-T distribution.
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
    metrics['R_recovery'] = 1 - np.linalg.norm(R_error) / np.linalg.norm(R_true) # measures how well R is recovered.

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
            label_map = {
                'RMSE_full': 'RMSE',
                'Bias_full': 'Bias',
                'Variance_full': 'Variance',
                'Std_full': 'Std',
                'Full_recovery': 'Recovery',
                'M_RMSE': 'RMSE',
                'M_Bias': 'Bias',
                'M_Variance': 'Variance',
                'M_Std': 'Std',
                'M_recovery': 'Recovery',
                'R_RMSE': 'RMSE',
                'R_Bias': 'Bias',
                'R_Variance': 'Variance',
                'R_Std': 'Std',
                'R_recovery': 'Recovery'
            }
            label = label_map.get(k, k)
            rows.append({'Section': section, 'Metric': label,
                         'Value': float(numpy.array(metrics[k].get()).ravel()[0])})
            
    df = (pd.DataFrame(rows)
          .set_index(['Section', 'Metric']))
    
    print(df.to_string(float_format='{:.6f}'.format))


device = 'cuda'
tensor_shape = (2, 2, 5, 5)
target_local_std = 2.0
df = 2
rank = 3
missing_fraction = 0.9
I, Omega, M_true, R_true, noise = gen_synthetic_tensor(tensor_shape, rank, missing_fraction, target_local_std, df, seed, device)
I = np.array(I)
M_true = np.array(M_true)
R_true = np.array(R_true)
signal = M_true + R_true

Omega_all = np.array(Omega)
train_mask = Omega_all & (np.random.rand(*tensor_shape) < 0.8) # 80% of observed entries used for training.
test_mask = Omega_all & ~train_mask # remaining 20% of observed entries used for testing.

Omega = train_mask 

rho_values = [10]
gamma_values = [20]

iter = 0
data = []

for rho, gamma in itertools.product(rho_values, gamma_values):
    params_test = {
            'lengthscaleU': [3, 3, 3, 3],
            'lengthscaleR': [1, 1, 1, 1],
            'varianceU': [1, 1, 1, 1],
            'varianceR': [1, 1, 1, 1],
            'tapering_range': 2,
            'd_MaternU': 3,
            'd_MaternR': 3,
            'R': 3,
            'rho': rho, # Try 10, 15, 20 - higher rho should encourage more global structure.
            'gamma': gamma, # Try 10, 20 - higher gamma should encourage more local structure.
            'maxiter': 30,
            'K0': 1,
            'epsilon': 1e-5,
            'seed': seed}
    X, Rtensor, M = glskf.GLSKF(I, Omega, **params_test)

    test_rmse = float(np.sqrt(np.mean((I[test_mask] - X[test_mask])**2)))
    test_recovery = float(1 - np.linalg.norm((I[test_mask] - X[test_mask]) / np.linalg.norm(I[test_mask])))
    test_error = float(np.linalg.norm(I[test_mask] - X[test_mask]) / np.linalg.norm(I[test_mask]))
    test_rmse_clean = float(np.sqrt(np.mean((signal[test_mask] - X[test_mask])**2)))
    test_recovery_clean = float(1 - np.linalg.norm((signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask])))
    test_error_clean = float(np.linalg.norm(signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask]))

    iter += 1
    data.append({'iteration': iter,
                 'rho': rho,
                 'gamma': gamma,
                 'test_rmse': test_rmse,
                 'test_recovery': test_recovery,
                 'test_error': test_error,
                 'test_rmse_clean': test_rmse_clean,
                 'test_recovery_clean': test_recovery_clean,
                 'test_error_clean': test_error_clean})

df = pd.DataFrame(data)
print(df.to_string())