import numpy as np
import torch
import cupy as cp
from scipy.ndimage import gaussian_filter
import pickle, os
import hashlib

# ========== Reproducibility ==========

def set_all_seeds(seed):
    np.random.seed(seed) # seeds stream draws M_true and R_true
    torch.manual_seed(seed) # seeds torch CPU stream, draws the noise
    cp.random.seed(seed) # seeds cupy's stream
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # seeds this GPU's stream, draws Omega
        torch.cuda.manual_seed_all(seed) # seeds all GPU streams if multiple GPUs are used

def gen_synthetic_tensor(shape, rank, missing_fraction, target_local_std,
                         noise_name, noise_params, seed, device):
    """
    Simple synthetic tensor generator with:
        Global structure: smooth low-rank structure.
        Local structure: spatial correlations.
        Heavy-tails: sparse extreme events.

    Args:
        shape (tuple): input tensor shape.
        rank (int): rank used in CP decomposition.
        missing_fraction (float): percentage of missing data entries.
        target_local_std (float): desired standard deviation of the local structure.
        noise_name (string): name of the noise distribution.
        noise_params (dict): parameters for the noise distribution.
        seed (int): ensures reproducibility.
        device (string): CPU or GPU performance.

    Returns:
        ndarray: synthetic tensor to be used in QKTF algorithm.
    """
    set_all_seeds(seed) # reset every RNG before drawing anything

    D = len(shape) # numer of tensor modes

    # ========== Global structure ==========
    def sigma_global(Id):
        return (min(10, max(Id / 4, 0.5))) # scales smoothing bandwidth with axis length, capped at 10, floor at 0.5

    M_true = torch.zeros(shape, device=device) # initialises the true M as a zero tensor

    for r in range(rank): # iterates over each rank component.
        factors = [] # stores each factor - used when generating one smooth factor per dimension.
        for d in range(D): # iterates over each dimension.
            u_d = np.random.randn(shape[d]) # random vector of length shape[d].
            u_d = gaussian_filter(u_d, sigma=sigma_global(shape[d])) # smooths the vector - creates global pattern.
            factors.append(u_d) # stores smoothed factor.

        # Compute D-dimensional outer product.
        component = factors[0] # starts with first factor - builds rank-one component.

        for d in range(1, D): # iteratively adds dimensions.
            component = component[..., None] # add new axis at end - broadcasts to next dim. '...' means all preceding dimensions.
            component = component * factors[d] # outer product with next factor.

        M_true += torch.tensor(component, dtype=torch.float32, device=device) # add this rank component to the true M.

    M_true_std = M_true.std().item()

    min_std = 1e-3
    if M_true_std < min_std:
        raise ValueError(
            f"M_true.std()={M_true_std:.2e} is below the safety floor ({min_std})."
            f"This shape/seed combination produced a degenerate global"
            f"structure - regenerate with a different seed of adjust sigma/shape"
        )

    M_true = M_true / M_true_std * 10 # normalise M to have a reasonable scale.

    # ========== Local structure ==========
    def sigma_local(Id, ratio=4.0, cap=4.0, floor=0.5):
        g = sigma_global(Id)
        return min(cap, max(g / ratio, floor))
    
    local_axis_sigma = [sigma_local(s) for s in shape]
    
    R_raw = np.random.randn(*shape)
    R_true = gaussian_filter(R_raw, sigma=local_axis_sigma) # short lengthscale vs sigma for global.
    R_true = R_true / R_true.std() * target_local_std
    R_true = torch.tensor(R_true, dtype=torch.float32, device=device)

    # ========== Heavy tails ==========
    dist = getattr(torch.distributions, noise_name)(**noise_params) # generates tensor with Cauchy distribution.
    noise = dist.sample(shape).to(device) # reshapes data to input tensor shape and sets to GPU performance.
    
    # ========== Tensor ==========
    tensor = M_true + R_true + noise # actual observed data.

    # ========== Mask creation ==========
    Omega = torch.rand(shape, device=device) >= missing_fraction # missing entries where random values are less than missing_fraction.

    return tensor, Omega, M_true, R_true, noise

def _config_hash(cfg):
    # encodes the dataset defining parameters into one string
    key = (f"{cfg.TENSOR_SHAPE}_{cfg.RANK}_{cfg.MISSING_FRACTION}_{cfg.TARGET_LOCAL_STD}"
           f"_{cfg.NOISE_NAME}_{sorted(cfg.NOISE_PARAMS.items())}")
    return hashlib.md5(key.encode()).hexdigest()[:10] # returns first 10 characters of the hash for brevity.

def get_or_create_tensor(seed, cfg, cache_dir="data/tensors"):
    """Generate once per seed, cache to disk, reload identically for every method."""
    os.makedirs(cache_dir, exist_ok=True) # avoids a crash if the cache folder doesn't exist.
    path = f"{cache_dir}/tensor_seed{seed}_{cfg.NOISE_NAME}_{_config_hash(cfg)}.pkl" # unqiue filename for each seed and config combination.

    if os.path.exists(path):
        # if the cache exists, reuse the exact same tensor instead of regenerating.
        with open(path, "rb") as f:
            d = pickle.load(f)

        I = cp.array(d["I"])
        Omega = cp.array(d["Omega"])
        M_true = cp.array(d["M_true"])
        R_true = cp.array(d["R_true"])
        noise = cp.array(d["noise"])

        return I, Omega, M_true, R_true, noise

    # if the cache doesn't exist, generate a new tensor and save it to disk.
    tensor, Omega, M_true, R_true, noise = gen_synthetic_tensor(
        cfg.TENSOR_SHAPE, cfg.RANK, cfg.MISSING_FRACTION, cfg.TARGET_LOCAL_STD,
        cfg.NOISE_NAME, cfg.NOISE_PARAMS, seed, cfg.DEVICE
    )
    
    I = cp.array(tensor)
    M_true = cp.array(M_true)
    R_true = cp.array(R_true)
    Omega = cp.array(Omega)
    noise = cp.array(noise)

    with open(path, "wb") as f: # saves so future calls hit the cache branch above.
        pickle.dump({"I": cp.asnumpy(I), "Omega": cp.asnumpy(Omega),
                     "M_true": cp.asnumpy(M_true), "R_true": cp.asnumpy(R_true), "noise": cp.asnumpy(noise)}, f)
    return I, Omega, M_true, R_true, noise
