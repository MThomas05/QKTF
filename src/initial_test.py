import qktf
import numpy
import torch
import cupy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

def gen_synthetic_tensor(shape, rank, missing_fraction, df, seed, device):
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
            u_d = np.random.randn(shape[d]) # random vector of length shape[d].
            u_d = gaussian_filter(u_d, sigma=5) # smooths the vector - creates global pattern.
            factors.append(u_d) # stores smoothed factor.

        # Compute D-dimensional outer product.
        component = factors[0] # starts with first factor - builds rank-one component.

        for d in range(1, D): # iteratively adds dimensions.
            component = component[..., None] # add new axis at end - broadcasts to next dim. '...' means all preceding dimensions.
            component = component * factors[d] # outer product with next factor.

    M_true += torch.tensor(component, device=device) # add this rank component to the true M.

    M_true = M_true / M_true.std() * 5 + 50 # normalise M to have a reasonable scale.

    # ========== Local structure ==========
    R_true = torch.randn(shape, device=device) * 2 # pure white noise.

    # ========== Heavy tails ==========
    noise = torch.distributions.StudentT(df=df, loc=0, scale=1) # generates tensor with Student-T distribution.
    dist = noise.sample(shape).to('cuda') # reshapes data to input tensor shape and sets to GPU performance.

    # ========== Tensor ==========
    tensor = M_true + R_true + dist # actual observed data.

    # ========== Mask creation ==========
    Omega = torch.rand(shape, device=device) >= missing_fraction # missing entries where random values are less than missing_fraction.

    return tensor, Omega, M_true, R_true, noise

params = {
    'lengthscaleU': [10, 10, 10],
    'lengthscaleR': [4, 4, 4],
    'varianceU': [1, 1, 1],
    'varianceR': [1, 1, 1],
    'tapering_range': 5,
    'd_MaternU': 3,
    'd_MaternR': 3,
    'R': 3,
    'psi': 0.001,
    'sigma': 0.001,
    'gamma': 0.001,
    'lambda_': 0.001,
    'tau': 0.5,
    'max_iter': 100,
    'K0': 50,
    'epsilon': 1e-4
}

if __name__ == "__main__":
    seed = 42
    device = 'cuda'
    tensor_shape = (50, 50, 30)
    df = 3
    rank = 3
    missing_fraction = 0.1
    I, Omega, M_true, R_true, noise = gen_synthetic_tensor(tensor_shape, rank, missing_fraction, seed, device)
    I = np.array(I)
    Omega = np.array(Omega)
    X, Rtensor, M = qktf.qktf(I, Omega, **params)

print(f"Original tensor: {I}")
print(f" X: {X}")
print(f"Rtensor: {Rtensor}")
print(f"M: {M}")