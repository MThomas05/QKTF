import cupy as np
import numpy
from tqdm import tqdm

def unfold(tensor, mode):
    """
    Performs mode-d unfolding of a tensor.

    Args:
        tensor (ndarray): input tensor to be unfolded.
        mode (int): the mode along which to unfold the tensor.

    Returns:
        ndarray: unfolded tensor.
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def fold(mat, dim, mode):
    """
    Performs mode-d folding of a matrix.

    Args:
        mat (ndarray): input matrix to be folded.
        dim (ndarray): 1d-array containing the dimensions of original tensor.
        mode (int): the mode along which to fold the matrix.

    Returns:
        ndarray: folded tensor.
    """
    index = list() # creates an empty list to store the new order of dimensions
    index.append(mode) # adds the mode to the index list
    for i in range(dim.shape[0]): # iterates through the axis of the dimension array
        if i != mode: # checks to ensure the current dimensions doesn't equal the mode
            index.append(i) # adds the current dimension to the index list
    return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

def shrinkage(X, t):
    U, Sig, VT = np.linalg.svd(X, full_matrices=False)
    Sig = np.maximum(Sig - t, 0)
    return (U * Sig) @ VT

def halrtc(I, Omega, alpha=None, rho0=1e-6, rho_mult=1.2, max_iter=200, tol=1e-5):
    N = I.shape
    D = len(N)
    if alpha is None:
        alpha = np.ones(D) / D

    train_vals = I[Omega > 0]
    mu = np.mean(train_vals)
    T = (I - mu) * Omega

    X = T.copy()
    Mi = [np.zeros(N) for _ in range(D)]
    Yi = [np.zeros(N) for _ in range(D)]
    rho = rho0

    train_norm = np.linalg.norm(T)
    last_X = X.copy()

    pbar = tqdm(total=max_iter, desc="HaLRTC Iterations")
    for k in range(max_iter):
        # Step 1: Mi update (SVT per mode-d unfolding)
        for d in range(D):
            mat = unfold(X + Yi[d] / rho, d)
            mat = shrinkage(mat, alpha[d] / rho)
            Mi[d] = fold(mat, np.array(N), d)

        # Step 2: X update, then project observed entries back to the mask
        X = sum(Mi[d] - Yi[d] / rho for d in range(D)) / D
        X = X * (1 - Omega) + T * Omega

        # Step 3: Yi update
        for d in range(D):
            Yi[d] = Yi[d] - rho * (Mi[d] - X)

        # Step 4: penalty continuation
        rho = rho_mult * rho

        # Convergence check
        rel_change = np.linalg.norm(X - last_X) / train_norm
        last_X = X.copy()
        pbar.update(1)
        if rel_change < tol:
            pbar.close()
            break
    else:
        pbar.close()

    return X + mu
