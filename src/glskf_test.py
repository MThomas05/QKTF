import glskf
import numpy
import torch
import cupy as np

def create_missing_mask(tensor_shape, missing_fraction, device):
    rand_tensor = torch.rand(tensor_shape, device=device) 
    mask = rand_tensor >= missing_fraction
    return mask

params = {
    'lengthscaleU': [10, 10, 10],
    'lengthscaleR': [2, 2, 2],
    'varianceU': [1, 1, 1],
    'varianceR': [1, 1, 1],
    'tapering_range': 5,
    'd_MaternU': 3,
    'd_MaternR': 3,
    'R': 3,
    'rho': 1,
    'gamma': 1,
    'maxiter': 100,
    'K0': 10,
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

    X, Rtensor, M = glskf.GLSKF(I, Omega_cp, **params)

print(f"Original tensor: {tensor}")
print(f" X: {X}")
print(f"M: {M}")
print(f"Rtensor: {Rtensor}")