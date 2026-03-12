import cupy as np
import numpy
import torch

numpy.random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

tensor = torch.rand((64, 64, 32))
Omega = (torch.rand(tensor.shape) < 0.1).float()
tensor_observed = tensor * Omega
tensor_obs = numpy.array(tensor_observed)
Omega = numpy.array(Omega)
tensor_obs_cupy = np.array(tensor_observed)
Omega_cupy = np.array(Omega)

if __name__ == "__main__":
    print("Testing QKTF on synthetic data...")

    params = {
        'lengthscaleU': [15, 15, 8], # Global smoothness
        'lengthscaleR': [4, 4, 2], # Local smoothness
        'varianceU': [1, 1, 1], # Global variance
        'varianceR': [1, 1, 1], # Local variance
        'tapering_range': 8, # Local kernel range
        'd_maternU': 3, # Matern degree
        'd_maternR': 3, # Matern degree
        'R': 20, # Rank
        'psi': 0.1, # Global regularisation
        'sigma': 20.0, # Global ADMM penalty
        'lambda_': 20.0, #Local ADMM penalty
        'tau': 0.5, # Quantile parameter
        'maxiter': 1000, # Maximum iterations
        'K0': 20, # When to start local component
        'epsilon': 1e-4, # Convergence tolerance
    }
    from qktf import qktf

    I_recovered, M_component, R_component = qktf(tensor_obs_cupy, Omega_cupy, **params)

    print(f"\n Algorithm Finished")
    print(f"Original tensor: {tensor}")
    print(f"I_recovery: {I_recovered}, Global component: {M_component}, Local component: {R_component}")
    