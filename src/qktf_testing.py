import qktf
import numpy
import torch
import cupy as np
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
import glskf

seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

pd.set_option('display.max_rows', None)

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
            u_d = gaussian_filter(u_d, sigma=10) # smooths the vector - creates global pattern.
            factors.append(u_d) # stores smoothed factor.

        # Compute D-dimensional outer product.
        component = factors[0] # starts with first factor - builds rank-one component.

        for d in range(1, D): # iteratively adds dimensions.
            component = component[..., None] # add new axis at end - broadcasts to next dim. '...' means all preceding dimensions.
            component = component * factors[d] # outer product with next factor.

        M_true += torch.tensor(component, dtype=torch.float32, device=device) # add this rank component to the true M.

    M_true = M_true / M_true.std() * 5 + 50 # normalise M to have a reasonable scale.

    # ========== Local structure ==========
    R_raw = numpy.random.randn(*shape)
    R_true = gaussian_filter(R_raw, sigma=2) # short lengthscale vs sigma for global.
    R_true = R_true / R_true.std() * target_local_std
    R_true = torch.tensor(R_true, dtype=torch.float32, device=device)

    # ========== Heavy tails ==========
    noise = torch.distributions.StudentT(df=df, loc=0.0, scale=1.0) # generates tensor with Student's t distribution.
    dist = noise.sample(shape).to(device) # reshapes data to input tensor shape and sets to GPU performance.
    
    # ========== Tensor ==========
    tensor = M_true + R_true + dist # actual observed data.

    # ========== Mask creation ==========
    Omega = torch.rand(shape, device=device) >= missing_fraction # missing entries where random values are less than missing_fraction.

    return tensor, Omega, M_true, R_true, dist

results = []

device = 'cuda'
tensor_shape = (2, 2, 5, 5)
target_local_std = 2.0
df = 2
rank = 3
missing_fraction = [0.9]
I, Omega, M_true, R_true, noise = gen_synthetic_tensor(tensor_shape, rank, missing_fraction, target_local_std, df, seed, device)
I = np.array(I)
M_true = np.array(M_true)
R_true = np.array(R_true)
signal = M_true + R_true

Omega_all = np.array(Omega)
train_mask = Omega_all & (np.random.rand(*tensor_shape) < 0.8) # 80% of observed entries used for training.
test_mask = Omega_all & ~train_mask # remaining 20% of observed entries used for testing.

Omega = train_mask 

psi_values = [0.001]
sigma_values = [0.05]
gamma_values = [0.001]
lambda_values = [0.05]
tau_values = [0.25, 0.5, 0.75]

iter = 0

data = []

for psi, sigma, gamma, lambda_, tau in itertools.product(psi_values, sigma_values, gamma_values, lambda_values, tau_values):
    params_test2 = {
                    'lengthscaleU': [3, 3, 3, 3], # 10-20% of dimension size for global structure.
                    'lengthscaleR': [1, 1, 1, 1], # 5-10% of dimension size for local structure.
                    'varianceU': [1, 1, 1, 1],
                    'varianceR': [1, 1, 1, 1],
                    'tapering_range': 2, # At least x2 the local lengthscale to capture local structure.
                    'd_MaternU': 3,
                    'd_MaternR': 3,
                    'R': 3,
                    'psi': psi,
                    'sigma': sigma,
                    'gamma': gamma, # Try 1e-5
                    'lambda_': lambda_, # Try 2.5e-7
                    'tau': tau, # Try [0.2, 1.0]
                'max_iter': 30,
                'K0': 5,
                'epsilon': 1e-5,
                'seed': seed}
    X, Rtensor, M = qktf.qktf(I, Omega, **params_test2)

    test_rmse = float(np.sqrt(np.mean((I[test_mask] - X[test_mask])**2)))
    test_recovery = float(1 - np.linalg.norm((I[test_mask] - X[test_mask]) / np.linalg.norm(I[test_mask])))
    test_error = float(np.linalg.norm(I[test_mask] - X[test_mask]) / np.linalg.norm(I[test_mask]))
    test_rmse_clean = float(np.sqrt(np.mean((signal[test_mask] - X[test_mask])**2)))
    test_recovery_clean = float(1 - np.linalg.norm((signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask])))
    test_error_clean = float(np.linalg.norm(signal[test_mask] - X[test_mask]) / np.linalg.norm(signal[test_mask]))
    iter += 1
    data.append({'iteration': iter,
                 'psi': psi,
                 'sigma': sigma,
                 'gamma': gamma,
                 'lambda_': lambda_,
                 'tau': tau,
                 'test_rmse': test_rmse,
                 'test_recovery': test_recovery,
                 'test_error': test_error,
                 'test_rmse_clean': test_rmse_clean,
                 'test_recovery_clean': test_recovery_clean,
                 'test_error_clean': test_error_clean})

df = pd.DataFrame(data)
print(df.to_string())