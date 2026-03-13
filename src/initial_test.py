import cupy as np
import matplotlib.pyplot as plt
import numpy
import torch
from qktf import qktf

# ========== Synthetic Data Generator ==========
def gaussian_smooth_3d(data, sigma=2.0):
    kernel_size = int(4 * sigma)
    x = numpy.arange(-kernel_size, kernel_size + 1)
    kernel = numpy.exp(-x**2  (2 * sigma**2))
    kernel = kernel / kernel.sum()

    smoothed = data.copy()
    
    for i in range(data.shape[0]):
        for k in range(data.shape[2]):
            smoothed[i, :, k] = numpy.convolve(data[i, :, k], kernel, mode='same')

    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            smoothed[:, j, k] = numpy.convolve(smoothed[:, j, k], kernel, mode='same')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            smoothed[i, j, :] = numpy.convolve(smoothed[i, j, :], kernel, mode='same')
    
    return smoothed

def synthetic_tensor_4d(rows=15, cols=20, depth=10, features=4, obs_rate=0.7, seed=42):
    numpy.random.seed(seed)

    # Global component
    global_pattern = numpy.zeros((rows, cols, depth, features))
    x = numpy.linspace(0, 2*numpy.pi, rows)
    y = numpy.linspace(0, 3*numpy.pi, cols)
    z = numpy.linspace(0, 2*numpy.pi, depth)
    X, Y, Z = numpy.meshgrid(y, x, z, indexing='xy')

    for f in range(features):
        freq_x = 1 + f*0.3
        freq_y = 1 + f*0.4
        freq_z = 1 + f*0.2

        pattern = (0.5 + 0.2*numpy.sin(freq_x*X) + 0.15*numpy.cos(freq_y*Y) + 0.1*numpy.sin(freq_z*Z))

        pattern_smooth = gaussian_smooth_3d(pattern, sigma=1.5)

        global_pattern[:, :, :, f] = pattern_smooth

    for f in range(features):
        feat_min = global_pattern[:, :, :, f].min()
        feat_max = global_pattern[:, :, :, f].max()
        global_pattern[:, :, :, f] = (global_pattern[:, :, :, f] - feat_min) / (feat_max - feat_min)


    # Local component
    local_pattern = numpy.zeros((rows, cols, depth, features))

    n_spikes = 15
    spike_locations = []

    for _ in range(n_spikes):
        i = numpy.random.randint(2, rows-2)
        j = numpy.random.randint(2, cols-2)
        k = numpy.random.randint(1, depth-1)
        spike_locations.append((i, j, k))

        for f in range(features):
            amplitude = numpy.random.uniform(0.2, 0.4)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        if (0 <= i+di < rows and
                            0 <= j+dj < cols and
                            0 <= k+dk < depth):
                            decay = numpy.exp(-0.5 * (di**2 + dj**2 + dk**2))
                            local_pattern[i+di, j+dj, k+dk, f] += amplitude * decay
    
    # Combine components
    tensor_clean = global_pattern + local_pattern
    tensor_noisy = tensor_clean + numpy.random.randn(rows, cols, depth, features) * 0.03
    tensor_final = (tensor_noisy - tensor_noisy.min()) / (tensor_noisy.max() - tensor_noisy.min())

    Omega = (numpy.random.rand(rows, cols, depth, features) < obs_rate).astype(float)

    global_gt = (global_pattern - global_pattern.min()) / (global_pattern.max() - global_pattern.min())
    local_gt = local_pattern / (tensor_noisy.max() - tensor_noisy.min())

    return {
        'tensor': torch.tensor(tensor_final, dtype=torch.float32),
        'Omega': torch.tensor(Omega, dtype=torch.float32),
        'global_gt': global_gt,
        'local_gt': local_gt,
        'spike_locations' : spike_locations,
        'shape': (rows, cols, depth, features),
        'description': f'Synthetic ({rows}x{cols}x{depth}x{features}) tensor: smooth 3d patterns + {n_spikes} 3d spikes'
    }

if __name__ == "__main__":

    print("QKTF SYNTHETIC TEST")

    # ========== Generate Synthetic Data ==========
    print("\n [1/4] Generating data")
    data = synthetic_tensor_4d(rows=15, cols=20, depth=10, features=4, obs_rate=0.7, seed=42)
    tensor = data['tensor']
    Omega = data['Omega']
    global_gt = data['global_gt']
    local_gt = data['local_gt']
    spike_locations = data['spike_locations']
    tensor_observed = tensor * Omega
    tensor_obs_cupy = np.array(tensor_observed.numpy())
    Omega_cupy = np.array(Omega.numpy())

    # ========== Run QKTF ==========
    print("\n [2/4] Running QKTF")
    params = {'lengthscaleU': [6, 8, 4, 3],
              'lengthscaleR': [1.5, 1.5, 1.5, 1.5],
              'varianceU': [1, 1, 1, 1],
              'varianceR': [1, 1, 1, 1],
              'tapering_range': 6,
              'd_maternU': 3,
              'd_maternR': 3,
              'R': 5,
              'psi': 0.01,
              'sigma': 0.05,
              'lambda_': 0.001,
              'gamma': 0.01,
              'tau': 0.5,
              'maxiter': 200,
              'K0': 10,
              'epsilon': 1e-4}
    
    I_recovered, M_component, R_component = qktf(tensor_obs_cupy, Omega_cupy, **params)

    # ========== Evaluation ==========
    print("\n [3/4] Evaluating results")

    I_true = tensor.numpy()
    I_recovered_np = np.asnumpy(I_recovered)
    M_np = np.asnumpy(M_component)
    R_np = np.asnumpy(R_component)
    Omega_np = Omega.numpy().astype(bool)

    obs_true = I_true[Omega_np]
    obs_recovered = I_recovered_np[Omega_np]
    train_rmse = numpy.sqrt(numpy.mean((obs_true - obs_recovered)**2))
    train_mse = numpy.mean((obs_true - obs_recovered)**2)
    train_bias = numpy.mean(obs_recovered - obs_true)

    missing_mask = ~Omega_np
    missing_true = I_true[missing_mask]
    missing_recovered = I_recovered_np[missing_mask]

    test_rmse = numpy.sqrt(numpy.mean((missing_true - missing_recovered)**2))
    test_mse = numpy.mean((missing_true - missing_recovered)**2)
    test_bias = numpy.mean(missing_recovered - missing_true)

    M_norm = numpy.linalg.norm(M_np)
    R_norm = numpy.linalg.norm(R_np)
    total_norm = numpy.linalg.norm(I_recovered_np)
    global_contr = (M_norm / total_norm * 100) if total_norm > 0 else 0
    local_contr = (R_norm / total_norm * 100) if total_norm > 0 else 0

    # ========== Performance Results ==========
    print("Performance Metrics")
    print("Training Metrics")

    print(f"test_rmse: {test_rmse:.4f}")
    print(f"test_mse: {test_mse:.4f}")
    print(f"test_bias: {test_bias:.4f}")

    print("Test Metrics")

    print(f"test_rmse: {test_rmse:.4f}")
    print(f"test_mse: {test_mse:.4f}")
    print(f"test_bias: {test_bias:.4f}")

    print("Contribution Analysis")

    print(f"global_contr: {global_contr:.4f}%")
    print(f"local_contr: {local_contr:.4f}%")

    print("Tensor Comparison")

    print(f"tensor: {I_true[0, :10]}")
    print(f"global: {M_component[0, :10]}")
    print(f"local: {R_component[0, :10]}")
    print(f"recovered: {I_recovered[0, :10]}")
    

    

    