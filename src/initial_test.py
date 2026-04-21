import qktf
import numpy
import torch
import cupy as np

def create_missing_mask(tensor_shape, missing_fraction, device):
    """
    Function that generates a mask for the input tensor

    Args:
        tensor_shape (tuple): input tensor shape
        missing_fraction (float): percentage of missing data entries
        device (string): CPU/GPU performance

    Returns:
        ndarray: Boolean mask with False for missing entries 
    """
    rand_tensor = torch.rand(tensor_shape, device=device) # generates random tensor of the same shape.
    mask = rand_tensor >= missing_fraction # missing entries where random values are less than missing_fraction.
    return mask

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
    torch.manual_seed(42)
    np.random.seed(42)
    tensor_shape = (50, 50, 30)
    df = 3
    dist = torch.distributions.StudentT(df=df, loc=0, scale=1) 
    tensor = dist.sample(tensor_shape).to('cuda')
    Omega = create_missing_mask(tensor_shape, missing_fraction=0.1, device='cuda')
    I = np.array(tensor)
    Omega_cp = np.array(Omega)
    num_obs = int(np.sum(Omega_cp))

    X, Rtensor, M = qktf.qktf(I, Omega_cp, **params)

print(f"Original tensor: {tensor}")
print(f" X: {X}")
print(f"Rtensor: {Rtensor}")
print(f"M: {M}")