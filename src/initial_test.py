import cupy as np
import numpy
import torch

tensor = torch.rand((25, 50))
Omega = (torch.rand(tensor.shape) < 0.7).float()
tensor_observed = tensor * Omega
tensor_obs = numpy.array(tensor_observed)
Omega = numpy.array(Omega)
tensor_obs_cupy = np.array(tensor_observed)
Omega_cupy = np.array(Omega)

if __name__ == "__main__":
    print("Testing QKTF on synthetic data...")

    params = {
        'lengthscaleU': [10, 5], # Global smoothness
        'lengthscaleR': [1, 3], # Local smoothness
        'varianceU': [1, 1], # Global variance
        'varianceR': [1, 1], # Local variance
        'tapering_range': 5, # Local kernel range
        'd_maternU': 3, # Matern degree
        'd_maternR': 3, # Matern degree
        'R': 3, # Rank
        'psi': 0.0001, # Global regularisation
        'sigma': 0.001, # Global ADMM penalty
        'lambda_': 0.1, #Local ADMM penalty
        'tau': 0.5, # Quantile parameter
        'maxiter': 100, # Maximum iterations
        'K0': 5, # When to start local component
        'epsilon': 1e-4, # Convergence tolerance
    }
    from qktf import qktf

    I_recovered, M_component, R_component = qktf(tensor_obs_cupy, Omega_cupy, **params)

    I_true = tensor.numpy()
    I_recovered_np = np.asnumpy(I_recovered)
    M_np = np.asnumpy(M_component)
    R_np = np.asnumpy(R_component)
    Omega_np = Omega.numpy().astype(bool)

    obs_true = I_true[Omega_np]
    obs_recovered = I_recovered_np[Omega_np]
    train_rmse = numpy.sqrt(numpy.mean((obs_true - obs_recovered) ** 2))
    train_mae = numpy.mean(numpy.abs(obs_true - obs_recovered))

    missing_mask = ~Omega_np
    missing_true = I_true[missing_mask]
    missing_recovered = I_recovered_np[missing_mask]

    test_rmse = numpy.sqrt(numpy.mean((missing_true - missing_recovered) ** 2))
    test_mae = numpy.mean(numpy.abs(missing_true - missing_recovered))
    test_bias = numpy.mean(missing_true - missing_recovered)

    M_norm = numpy.linalg.norm(M_np)
    R_norm = numpy.linalg.norm(R_np)
    total_norm = numpy.linalg.norm(I_recovered_np)

    global_contr = (M_norm / total_norm * 100) if total_norm > 0 else 0
    local_contr = (R_norm / total_norm * 100) if total_norm > 0 else 0

    naive_pred = numpy.mean(obs_true)
    naive_rmse = numpy.sqrt(numpy.mean((missing_true - naive_pred) ** 2))
    improvement = ((naive_rmse - test_rmse) / naive_rmse * 100)

    print(f"\n TEST ERROR (Missing Entries):")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    print(f"BIAS: {test_bias:.6f}")

    print(f"\n TRAINING ERROR (Observed Entries)")
    print(f"RMSE: {train_rmse:.6f}")
    print(f"MAE: {train_mae:.6f}")

    print(f"\n CONTRIBUTION ANALYSIS:")
    print(f"Global: {global_contr:.1f}%")
    print(f"Local: {local_contr:.1f}%")
    print(f"M Norm: {M_norm:.4e}")
    print(f"R Norm: {R_norm:.4e}")

    print(f"\n Algorithm Finished")
    print(f"Original tensor: {tensor}")
    print(f"I_recovery: {I_recovered}, Global component: {M_component}, Local component: {R_component}")
    