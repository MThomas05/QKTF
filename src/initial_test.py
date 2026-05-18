import qktf
import numpy
import torch
import cupy as np
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter

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
    R_raw = numpy.random.standard_t(df=df, size=shape) # short lengthscale vs sigma for global.
    R_true = median_filter(R_raw, size=3)
    R_true = R_true / numpy.std(R_true) * target_local_std 
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
    metrics['R_recovery'] = 1 - np.linalg.norm(R_error) / np.linalg.norm(R_true) # measures how well R is recovered.

    return metrics

def print_diagnostics(metrics):
    sections = {
        'Full': ['RMSE_full', 'Bias_full', 'Variance_full', 'Std_full'],
        'Global M': ['M_RMSE', 'M_Bias', 'M_Variance', 'M_Std'],
        'Local R': ['R_RMSE', 'R_Bias', 'R_Variance', 'R_Std']
    }
    rows = []
    for section, keys in sections.items():
        for k in keys:
            label = k.split('_', 1)[-1] if '_' in k else k
            rows.append({'Section': section, 'Metric': label,
                         'Value': float(np.array(metrics[k].get()).ravel()[0])})
            
    df = (pd.DataFrame(rows)
          .set_index(['Section', 'Metric']))
    
    print(df.to_string(float_format='{:.6f}'.format))

results = []

seed = 3
device = 'cuda'
tensor_shape = (25, 25, 30, 30)
target_local_std = 5.0
df = 2
rank = 4
missing_fraction = 0.2
I, Omega, M_true, R_true, noise = gen_synthetic_tensor(tensor_shape, rank, missing_fraction, target_local_std, df, seed, device)
I = np.array(I)
Omega = np.array(Omega)
M_true = np.array(M_true)
R_true = np.array(R_true)

n_obs = int(np.sum(Omega))
lambda_base = target_local_std / n_obs

psi_values = [0.002, 0.001]
sigma_values = [0.01, 0.02]
gamma_values = [5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1.0]
lambda_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]

test_iter = 0

for psi, sigma, gamma, lam in itertools.product(psi_values, sigma_values, gamma_values, lambda_values):
    params_test = {
                   'lengthscaleU': [5, 5, 5, 5], # 10-20% of dimension size for global structure.
                   'lengthscaleR': [2, 2, 2, 2], # 5-10% of dimension size for local structure.
                   'varianceU': [1, 1, 1, 1],
                   'varianceR': [1, 1, 1, 1],
                   'tapering_range': 4, # At least x2 the local lengthscale to capture local structure.
                   'd_MaternU': 3,
                   'd_MaternR': 3,
                   'R': 3,
                   'psi': psi, # Try 0.003, 0.004, 0.005
                   'sigma': sigma, # Try 0.03, 0.04
                   'gamma': gamma, # Try 0.0001, 0.0002
                   'lambda_': lam, # Try 0.02, 0.03, 0.04, 0.05
                   'tau': 0.5,
                   'max_iter': 50,
                   'K0': 10,
                   'epsilon': 1e-8,
                   'seed': seed}
    X, Rtensor, M = qktf.qktf(I, Omega, **params_test)

    m_std = float(np.std(M))
    r_std = float(np.std(Rtensor))

    results.append({
        'psi': psi, 'sigma': sigma, 'gamma': gamma, 'lambda_': lam,
        'M_std': m_std, 'R_std': r_std
    })

    print(f"psi={psi:.6f}, sigma={sigma:.6f}, gamma={gamma:.6f}, lambda_={lam:.6f}",
          f"M_std={m_std}, R_std={r_std}")
    
    test_iter += 1
    data = pd.DataFrame(results)
    data['score'] = (data['M_std'] - 5).abs() - (data['R_std'] - 5).abs()
    mask = data['M_std'].between(3.5, 7.5) & data['R_std'].between(3.5, 7.5)
    print(data.loc[mask].sort_values('score'))
    print(f"iteration: {test_iter} out of 128")

metrics = compute_diagnostics(I, Omega, X, M_true, R_true, M, Rtensor)
print_diagnostics(metrics)