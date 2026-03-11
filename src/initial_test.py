import cupy as np
import numpy
import torch

tensor = torch.tensor([[[1, 2], [3, 4]], [[2, 1], [4, 3]]], dtype=torch.float32)
Omega = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.float32)
tensor_observed = tensor * Omega
tensor_obs_np = numpy.array(tensor_observed)
Omega_np = numpy.array(Omega)
tensor_obs_cupy = np.array(tensor_observed)
Omega_cupy = np.array(Omega)

if __name__ == "__main__":
    print("Testing QKTF on synthetic data...")

    params = {
        'lengthscaleU': [1, 1, 1], # Global smoothness
        'lengthscaleR': [1, 1, 1], # Local smoothness
        'varianceU': [1, 1, 1], # Global variance
        'varianceR': [1, 1, 1], # Local variance
        'tapering_range': 1, # Local kernel range
        'd_maternU': 3, # Matern degree
        'd_maternR': 3, # Matern degree
        'R': 1, # Rank
        'psi': 1.0, # Global regularisation
        'sigma': 1.0, # Global ADMM penalty
        'lambda_': 1.0, #Local ADMM penalty
        'tau': 0.5, # Quantile parameter
        'maxiter': 100, # Maximum iterations
        'K0': 10, # When to start local component
        'epsilon': 1e-4, # Convergence tolerance
        'verbose': True, # Print progress
    }
    from qktf import qktf

    I_recovered, M_component, R_component, info = qktf(tensor_obs_cupy, Omega_cupy, **params)

    print(f"\n Algorithm Finished")
    