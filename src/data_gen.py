import numpy, torch, cupy as np
from scipy.ndimage import gaussian_filter
import pickle, os
import hashlib

def set_all_seeds(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def gen_synthetic_tensor(shape, rank, missing_fraction, target_local_std, df, seed, device):
    set_all_seeds(seed)
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

def _config_hash(cfg):
    key = f"{cfg.TENSOR_SHAPE}_{cfg.RANK}_{cfg.MISSING_FRACTION}_{cfg.TARGET_LOCAL_STD}_{cfg.DF}"
    return hashlib.md5(key.encode()).hexdigest()[:10]

def get_or_create_tensor(seed, cfg, cache_dir="data/tensors"):
    """Generate once per seed, cache to disk, reload identically for every method."""
    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/tensor_seed{seed}_{_config_hash(cfg)}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        I = np.array(d["I"]); Omega_all = np.array(d["Omega_all"])
        M_true = np.array(d["M_true"]); R_true = np.array(d["R_true"])
        train_mask = np.array(d["train_mask"]); test_mask = np.array(d["test_mask"])
        return I, Omega_all, M_true, R_true, train_mask, test_mask

    tensor, Omega, M_true, R_true, noise = gen_synthetic_tensor(
        cfg.TENSOR_SHAPE, cfg.RANK, cfg.MISSING_FRACTION, cfg.TARGET_LOCAL_STD, cfg.DF, seed, cfg.DEVICE
    )
    I = np.array(tensor); M_true = np.array(M_true); R_true = np.array(R_true)
    Omega_all = np.array(Omega)

    set_all_seeds(seed)  # reset again so the train/test split is deterministic too
    train_mask = Omega_all & (np.random.rand(*cfg.TENSOR_SHAPE) < 0.8)
    test_mask = Omega_all & ~train_mask

    with open(path, "wb") as f:
        pickle.dump({"I": np.asnumpy(I), "Omega_all": np.asnumpy(Omega_all),
                     "M_true": np.asnumpy(M_true), "R_true": np.asnumpy(R_true),
                     "train_mask": np.asnumpy(train_mask), "test_mask": np.asnumpy(test_mask)}, f)
    return I, Omega_all, M_true, R_true, train_mask, test_mask
