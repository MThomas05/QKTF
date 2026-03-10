import cupy as np
import numpy

def test_tensor(shape=(50, 60, 30), missing_rate=0.3):
    np.random.seed(0)

    D = len(shape)
    I_true = np.zeros(shape)

    if D == 2:
        x = np.linspae(0, 4*np.pi, shape[0])
        y = np.linspace(0, 4*np.pi, shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        I_true = 50 + 20*np.sin(X) + 15*np.cos(Y) + 10*np.sin(X)*np.cos(Y)

    elif D == 3:
        x = np.linspace(0, 4*np.pi, shape[0])
        y = np.linspace(0, 4*np.pi, shape[1])
        z = np.linspace(0, 4*np.pi, shape[2])

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    I_true[i, j, k] = (50 + 20*np.sin(x[i]) + 15*np.cos(y[j]) + 10*np.sin(z[k]) + 5*np.sin(x[i])*np.cos(y[j]))
    
    else:
        I_true = np.random.randn(*shape)*10 + 50

    I_true = np.abs(I_true) + 1

    Omega = (np.random.rand(*shape) > missing_rate).astype(float)
    I_observed = I_true * Omega

    return I_true, I_observed, Omega    

if __name__ == "__main__":
    print("Testing QKTF on synthetic data...")

    print("\n[1/4]Generating synthetic tensor...")
    I_true, I_observed, Omega = test_tensor(shape=(50, 60, 30), missing_rate=0.3)

    print("\n[2/4] Setting Hyperparameters...")
    D = len(I_true.shape)

    params = {
        'lengthscaleU': [10, 10, 5], # Global smoothness
        'lengthscaleR': [3, 3, 2], # Local smoothness
        'varianceU': [1, 1, 1], # Global variance
        'varianceR': [1, 1, 1], # Local variance
        'tapering_range': 10, # Local kernel range
        'd_maternU': 3, # Matern degree
        'd_maternR': 3, # Matern degree
        'R': 10, # Rank
        'psi': 1.0, # Global regularisation
        'sigma': 1.0, # Global ADMM penalty
        'lambda_': 1.0, #Local ADMM penalty
        'tau': 0.5, # Quantile parameter
        'maxiter': 100, # Maximum iterations
        'K0': 10, # When to start local component
        'epsilon': 1e-4, # Convergence tolerance
        'verbose': True, # Print progress
    }

    print("Hyperparameters set:")

    print("\n[3/4]Running QKTF...")
    from qktf import qktf

    I_recovered, M_component, R_component, info = qktf(I_observed, Omega, **params)

    print(f"\n Algorithm Finished")
    