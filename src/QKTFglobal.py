import cupy as cp
import numpy
from tqdm import tqdm
from cupyx.scipy.linalg import khatri_rao
from cupyx.scipy.sparse import linalg, eye, csr_matrix

def cov_matern(d, loghyper, x):
    """
    Computes the Matern covariance matrix for a given dimension.
    """
    ell = cp.exp(loghyper[0])
    sf2 = cp.exp(2*loghyper[1])
    def f(t):
        if d == 1: return 1
        if d == 3: return 1 + t
        if d == 5: return 1 + t*(1 + t/3)
        if d == 7: return 1 + t*(1 + t*(6 + t)/15)
    def m(t):
        return f(t)*cp.exp(-t)
    dist_sq = ((x[:, None] - x[None, :])/ell)**2
    return sf2*m(cp.sqrt(d*dist_sq))

def bohman(loghyper, x):
    """
    Compute the Bohman taper.
    """
    range_ = cp.exp(loghyper[0])
    dis = cp.abs(x[:, None] - x[None, :])
    r = cp.minimum(dis/range_, 1)
    k = (1 - r)*cp.cos(cp.pi*r) + cp.sin(cp.pi*r)/cp.pi
    k[k < 1e-16] = 0
    k[cp.isnan(k)] = 0
    return k

def unfold(tensor, mode):
    """
    Performs mode-d unfolding of a tensor.

    Args:
        tensor (ndarray): input tensor to be unfolded.
        mode (int): the mode along which to unfold the tensor.

    Returns:
        ndarray: unfolded tensor.
    """
    return cp.reshape(cp.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

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
    index = [mode] + [i for i in range(dim.shape[0]) if i != mode]
    return cp.moveaxis(cp.reshape(mat, list(dim[index]), order='F'), 0, mode)

def build_khatri_rao(U, dims):
    """
    Builds the Khatri-Rao product of a list of matrices, all dimensions except the current one.

    Args:
        U (list): list of D latent matrices, where D is the number of dimensions of the input tensor.
        dims (ndarray): current dimension being updated in the ADMM iterations of the QKTF algorithm.

    Returns:
        ndarray: Khatri-Rao product of the list of matrices.
    """
    dims = [int(d) for d in dims] # sets D as the number of dimensions of the input tensor, which is equal to the length of the list of latent matrices.
    if len(dims) == 1:
        return U[dims[0]]
    else:
        result = U[dims[-1]]
        for i in range(len(dims) - 2, -1, -1):
            result = khatri_rao(result, U[dims[i]])
        return result

def reconstruct_tensor(U, shape):
    """
    Reconstructs the global component of the tensor from CP decomposition of the latent matrices.

    Args:
        U (list): list of D latent matrices, where D is the number of dimensions of the input tensor.
        shape (tuple): shape of the original tensor.

    Returns:
        ndarray: reconstructed global component of the tensor.
    """
    D = len(shape) # gets the number of dimensions.
    dims_except_0 = list(range(1, D)) # creates a list of dimensions except the first dimension - used for mode-0 unfolding.
    if len(dims_except_0) > 0: # checks to ensure there's more than one dimension - if not, the global component is just the first latent matrix.
        KrU = build_khatri_rao(U, dims_except_0) # builds the Khatri-Rao product of the latent matrices, excluding the first dimension.
        M_unfold = U[0] @ KrU.T # computes the mode-0 unfolding of the global component using the first latent matrix and the Khatri-Rao product.  
    else:
        M_unfold = U[0] # if there's only one dimension, the global component is just the first latent matrix.

    M = M_unfold.reshape(shape, order = 'F') # reshapes the mode-0 unfolding of the global component to match the original tensor shape.
    return M

def prox_map(xi, alpha, tau):
    """
    Proximal operator for the z-update step of the ADMM algorithm in the QKTF algorithm.

    Args:
        xi (ndarray): input vector for the proximal operator.
        alpha (float): parameter for the proximal operator - |Omega|*sigma.
        tau (float): quantile parameter for the ADMM algorithm.

    Returns:
        ndarray: output vector after applying the proximal operator.
    """
    low = (tau - 1)/alpha # calculates the lower bound for the proximal operator.
    high = tau/alpha # calculates the upper bound for the proximal operator.
    return xi - cp.maximum((tau - 1)/alpha, cp.minimum(xi, tau/alpha)) # applies the proximal operator to the input vector.

def global_operator(vec, maskT, KrU, KrU_T, Qu, psi, sigma, R, M):
    """
    Constructs the linear operator used in the global ADMM optimisation steps of the QKTF algorithm.

    Args:
        vec (ndarray): vector to be multiplied by the global operator.
        maskT (ndarray): boolean array indicating the observed entries of the tensor.
        KrU (ndarray): Khatri-Rao product of the latent matrices.
        KrU_T (ndarray): transpose of the Khatri-Rao product of the latent matrices.
        Qu (ndarray): covariance matrix used for covariance tapering in dimension d.
        psi (float):  smoothness parameter for covariance tapering.
        sigma (float): ADMM penalty parameter.
        R, M (int):

    Returns:
        ndarray: linear operator used in the Conjugate Gradient method for the global ADMM optimisation steps of the QKTF algorithm.
    """
    X = vec.reshape(R, M, order = 'F') # reshapes vector to match the dimension of fixed tensor
    temp = KrU @ X # computes the left-hand side product of Khatri-Rao product and the reshaped vector
    temp *= maskT # applies the mask through right-hand side multiplication - zeroes out the unobserved entries
    Ap1 = sigma * (KrU_T @ temp) # computes the first part of the linear operator - sigma*(H_d^T*O_d'^T*O_d'*H_d)
    Ap2 = (psi / (R * M)) * (X @ Qu) # computes the second part of the linear operator - psi*(K_d^u)^{-1})
    return (Ap1 + Ap2).ravel(order = 'F')

def global_admm(Qu, KrU, mask_matrixT, YR_tilde, priorvalue, z, theta,
                psi, sigma, inner_maxiter, tau, R, sum_obs, cg_maxiter, rel_tol, abs_tol, verbose=False):
    """
    Global ADMM algorithm for updating the latent matrices in the QKTF algorithm.

    Args:
        Qu (ndarray): covariance matrix used for covariance tapering in dimension d.
        KrU (ndarray): Khatri-Rao product of the latent matrices.
        mask_matrixT (ndarray): transpose of the boolean array indicating the observed entries of the tensor.
        YR_tilde (ndarray): fixed tensor vec(G_(d)^T)
        priorvalue (ndarray): previous iteration of latent matrix as first guess for algorithm.
        z, theta (ndarray): auxiliary and Lagrange multiplier variables for the ADMM algorithm.
        psi (float): smoothness parameter for covariance tapering.
        sigma (float): ADMM penalty parameter.
        max_iter (int): maximum number of iterations for the ADMM algorithm.
        tau (float): quantile parameter for ADMM algorithm.
        sum_obs (int): number of observed entries in the tensor.
        total_data (int): total number of entries in the tensor.

    Returns:
        r_vec (ndarray): updated latent matrix after the global ADMM optimisation steps of the QKTF algorithm.
        a_vec (ndarray): auxiliary variable after the global ADMM optimisation steps of the QKTF algorithm.
        v_vec (ndarray): Lagrange multiplier variable after the global ADMM optimisation steps of the QKTF algorithm.
    """
    M = YR_tilde.shape[1] # represents the shape of H_d^T*O_d'^T*O_d'*vec(G_(d)^T) which is size RI_d.
    x0 = priorvalue.copy() # sets the initial guess for the ADMM algorithm as the previous iteration of the latent matrix.
    KrU_T = KrU.T # computes the transpose of the Khatri-Rao product of the latent matrices.
    dtype = YR_tilde.dtype

    def matvec(vec): # performs y = Ax for the linear operator used in the Conjugate Gradient method.
        return global_operator(vec, mask_matrixT, KrU, KrU_T, Qu, psi, sigma, R, M) # returns the linear operator used in the Conjugate Gradient method.


    A = linalg.LinearOperator((R*M, R*M), matvec=matvec, dtype=dtype) # creates a linear operator for the Conjugate Gradient method, using the matvec function defined above.
    assert inner_maxiter > 0, "global_admm requires at least one ADMM sweep"
    # ========== ADMM iterations ==========
    converged = False
    gcg_nonconverged = 0
    for j in range(inner_maxiter):
        z_prev = z.copy() # stores the previous value of the auxiliary variable for convergence checking.
        u_prev = x0.copy()
        bmat = sigma * (YR_tilde - z) - theta # computes inside the bracket of 'b' - used in the Conjugate Gradient method.
        bmat = KrU_T @ (mask_matrixT * bmat)
        b = bmat.ravel(order='F')
        # u-update using Conjugate Gradient method.
        u, info = linalg.cg(A, b, x0=x0, atol=1e-4, maxiter=cg_maxiter) # performs the Conjugate Gradient method to solve vec(u).

        if info != 0:
            gcg_nonconverged += 1
        if verbose:
            print(f"[global_admm] sweep {j}: CG info={info}")


        x0 = u # warm-start the next ADMM iteration from this solution.
        umat = u.reshape(R, M, order = 'F') # reshapes the solution of the Conjugate Gradient method to match the dimension of the fixed tensor.
        temp = KrU @ umat # computes the H_d*vec(u) product.
        temp = mask_matrixT * temp # applies the mask.

        # z-update using Proximal operator.

        eta = YR_tilde - (theta / sigma) - temp # computes the input for the proximal operator.
        alpha = sum_obs * sigma # computes the alpha parameter for the proximal operator.
        z = prox_map(eta, alpha, tau) # applies the proximal operator to update the auxiliary variable.

        # theta-update.

        theta = theta + sigma * (temp + z - YR_tilde) # updates the Lagrange multiplier variable.

        if verbose:
            print(f"[global_admm] sweep {j}: z_norm={cp.linalg.norm(z)}, u_norm={cp.linalg.norm(u)}, theta_norm={cp.linalg.norm(theta)}")

        # convergence criterion.
        res_pri = temp + z - YR_tilde # computes the primal residual for convergence checking.
        res_temp = mask_matrixT * (z - z_prev)
        res_temp = KrU_T @ res_temp
        res_dual = sigma * res_temp # computes the dual residual for convergence checking.
        eps_pri = cp.sqrt(sum_obs) * abs_tol + rel_tol * max(cp.linalg.norm(temp), cp.linalg.norm(z), cp.linalg.norm(YR_tilde)) # computes the primal feasibility tolerance.
        eps_dual = cp.sqrt(R * M) * abs_tol + rel_tol * cp.linalg.norm(KrU_T @ theta) # computes the dual feasibility tolerance.
        
        if cp.linalg.norm(res_pri) <= eps_pri and cp.linalg.norm(res_dual) <= eps_dual: # checks for convergence of the ADMM algorithm.
            converged = True
            break

    return u, z, theta, j + 1, converged, gcg_nonconverged

def QKTFglobal(I, Omega, lengthscaleU: list, varianceU: list,
         d_MaternU, R, psi, sigma, tau, max_iter,
         epsilon, inner_maxiter, cg_maxiter=500, verbose=False, seed=None):
    """
    Quantized Kernelized Tensor Factorization (QKTF) algorithm for tensor completion.  

    Args:
        I (ndarray): input data tensor.
        Omega (ndarray): binary mask - same shape as I.
        lengthscaleU (list): list of lengthscales for the global covariance tapering in each dimension.
        varianceU (list): list of variances for the global covariance tapering in each dimension.
        tapering_range (float): range parameter for the global covariance tapering.
        d_maternU (float): degree of Matern kernel for global covariance tapering.
        R (int): CP decomposition rank used in reconstruction of global component.
        psi (float): smoothness parameter for covariance tapering.
        sigma (float): ADMM penalty parameter.
        tau (float): quantile parameter for ADMM algorithm.
        max_iter (int): maximum number of iterations for the ADMM algorithm.
        epsilon (float): convergence threshold for the ADMM algorithm.

    Returns:
        M_component (ndarray): reconstructed global component of the tensor.
    """
    # ========== Setup ==========
    N = I.shape # sets N as the shape of the input tensor.
    N = numpy.array(N) # converts N to a numpy array - created using NumPy and not CuPy as it's used for integer indexing. CuPy arrays cannot be used for integer indexing.
    
    D = I.ndim # sets D as the number of dimensions of the input tensor.

    # Assert inputs
    assert len(lengthscaleU) == D # ensures the number of lengthscales provided matches the number of dimensions of the input tensor.
    assert len(varianceU) == D # ensures the number of variances provided matches the number of dimensions of the input tensor.
    assert I.shape == Omega.shape # ensures the input tensor and the binary mask have the same shape.
    assert R > 0 # ensures the CP decomposition rank is a positive integer.
    assert 0 < tau < 1 # ensures the quantile parameter is between 0 and 1.

    # ========== Pre-processing data ==========

    # Binary indicator matrix
    Omega = Omega.astype(bool) # converts the binary mask to a boolean array - done due to memory efficiency (smaller than index arrays) and avoids explicit loops.
    pos_miss = cp.where(Omega == 0) # creates a tuple of arrays containing the indices of the missing entries in the tensor - can be used directly for indexing and can be unpacked correctly.
    num_obs = int(numpy.sum(Omega)) # calculates the number of observed entries in the tensor.
    total_data = int(numpy.prod(N)) # calculates the total number of entries in the tensor.

    # Mask construction
    mask_matrix = [unfold(Omega, d) for d in range(D)] # creates a list of D matrices, where each matrix is the mode-d unfolding of Omega.
    mask_matrixT = [mask_matrix[d].T for d in range(D)] # creates a list of D matrices, where each matrix is the transpose of the mode-d unfolding of Omega.
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)] # creates a list of D vectors, where each vector is the flattened version of the mode-d unfolding of Omega.
    pos_obs = [cp.where(mask_flat[d] == 1) for d in range(D)] # creates a list of D arrays, containing arrays of observed entries.

    # Data centering
    idx = cp.sum(mask_matrix[D-1], axis = 0) > 0 # creates a Boolean mask identifying which columns have at least one observed entry.
    train_matrix = I[Omega] # creates a mask of the tensor - setting indices to zero where there is data missing.
    centre = cp.mean(train_matrix)
    Isubmean = I - centre # centers the data by subtracting the mean of the observed entries from all entries in the tensor.

    T = Isubmean * Omega # creates a tensor of the centered observed entries - setting indices to zero where there is data missing.

    # ========== Building covariance matrices ==========
    hyper_Ku = [None] * D # creates an empty list to store the hyperparameters for the global and local covariance tapering, list length = D.
    Ku = [None] * D # creates an empty list to store the covariance matrices for the global and local covariance tapering, list length = D.
    inv_Ku = [None] * D # creates an empty list to store the inverse covariance matrices for the global covariance tapering, list length = D.

    for d in range(D-1): # iterates through each dimension of the input tensor.
        x = cp.arange(1, N[d] + 1) # creates a vector of integers from 1 to the size of the current dimension - used as input for the covariance function.

        # Global covariance
        hyper_Ku[d] = [cp.log(lengthscaleU[d]), cp.log(varianceU[d])] # sets the dth dimension of hyperparameters as log of lengthscale and log of variance.
        Ku[d] = cov_matern(d_MaternU, hyper_Ku[d], x) # computes the covariance matrix for the dth dimension using the Matern covariance function.
        inv_Ku[d] = cp.linalg.inv(Ku[d]) # inverts the covariance matrix for the dth dimension - used in the global ADMM optimisation steps of the QKTF algorithm.

    inv_Ku[D-1] = cp.eye(N[D-1]) # intialises the global covariance in the last dimension as identity matrix.

    # ========== Initialisation for ADMM iterations ==========
    X = T.copy() # sets the initial value of the fixed tensor as the centered observed entries.
    X[pos_miss] = T.sum() / num_obs # sets the missing entries of the fixed tensor as the mean of the observed entries.

    rng = numpy.random.default_rng(seed)

    z, theta = [], []
    for d in range(D):
        dims = [N[i] for i in range(D) if i != d]
        unfold_shape = (int(cp.prod(cp.array(dims))), N[d])
        z.append(cp.zeros(unfold_shape))
        theta.append(cp.zeros(unfold_shape))
    U = [cp.asarray(rng.standard_normal((N[d], R))) for d in range(D)] # intialises the latent matrices as random values from a standard Gaussian distribution, scaled by 0.1 to ensure no crashing.
    M = reconstruct_tensor(U, N) # intial reconstruction of M.
    Uvector = [U[d].ravel(order = 'F') for d in range(D)] # creates a list of D vectors, where each vector is the flattened version of the corresponding latent matrix.
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)] # creates a list of D vectors, where each vector is the flattened version of the transpose of the corresponding latent matrix.
    X[pos_miss] = M[pos_miss] # sets the missing entries of X to the sum of the missing entries of global and local components.

    d_all = cp.arange(D) # creates a vector of integers from 0 to D-1 - used for indexing.
    train_norm = cp.linalg.norm(T) # calculates the norm of the tensor of the centered observed entries - used for convergence checking.
    last_ten = T.copy() # initialises a tensor to store the value of the fixed tensor from the previous iteration for convergence checking.
    pbar = tqdm(total=max_iter, desc="QKTFglobal Iterations") # creates a progress bar for the ADMM iterations.
    iter = 0 # initialises the iteration counter for the ADMM algorithm.

    global_sweep_history = [] 
    global_hit_cap = 0
    global_cg_nonconverged = 0

    while True: # runs the ADMM iterations until the maximum number of iterations is reached.
        Gtensor = X # initialises the global component of the tensor as the initial fixed tensor minus the local tensor.
        Gtensor_mask = Gtensor * Omega # masks the global tensor - setting indices to zero where there is data missing.

        # Global component iteration
        global_sweeps = []

        for d in range(D): # iterates through each dimension of the input tensor.
            dsub = cp.delete(d_all, d) # deletes dth dimension from list.
            dsub = cp.array(dsub) # creatse an array of dsub.
            Gtensor_unfold = unfold(Gtensor_mask, d).T # unfolds the masked global tensor along the current dimension - creates O_d'*vec(G_(d)^T) - now has size |Omega|.
            KrU = build_khatri_rao(U, dsub) # builds the Khatri-Rao product of the latent matrices, excluding the current dimension - creates H_d.

            # Actual Global ADMM optimisation call.
            UTvector[d], z[d], theta[d], g_sweeps, g_converged, gcg_nonconverged = global_admm(
                inv_Ku[d], KrU, mask_matrixT[d], Gtensor_unfold,
                UTvector[d], z[d], theta[d], psi, sigma, inner_maxiter, tau, R,
                num_obs, cg_maxiter=cg_maxiter, rel_tol=1e-4, abs_tol=1e-4, verbose=verbose
                )
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T # reshapes the latent matrix back to its original shape.)

            global_sweep_history.append(g_sweeps)
            global_sweeps.append(g_sweeps)
            if not g_converged:
                global_hit_cap += 1
            global_cg_nonconverged += gcg_nonconverged
        
        M = reconstruct_tensor(U, N) # reconstructs the global component of the tensor from the CP decomposition of the latent matrices.

        X[pos_miss] = M[pos_miss] # updates the missing entries of the fixed tensor as the sum of the global component and the local tensor.
        Xori = X + centre # adds the mean back into X.

        print(f"[QKTFglobal] M_norm={cp.linalg.norm(M + centre)}",
              f"[QKTFglobal] Xori_norm={cp.linalg.norm(Xori)}")
        
        # Convergence checks
        iter += 1 # increments the iteration counter.
        tol = cp.linalg.norm((X - last_ten)) / train_norm # calculates the convergence metric as the relative change in the fixed tensor.
        last_ten = X.copy() # updates the tensor for convergence checking to the current fixed tensor.

        pbar.update(1)
        pbar.set_postfix({
            'tol': f'{tol:.2e}',
            'g_sweeps (max)': int(max(global_sweeps)),
            'g_hit_cap': sum(1 for s in global_sweeps if s >= max_iter),
        })
        
        if cp.isnan(tol) or cp.isinf(tol): # checks for numerical issues in convergence metric.
            pbar.set_postfix({'tol': f'{tol:.2e}', 'epoch': iter})
            break
        
        if (tol < epsilon and iter > 1) or (iter >= max_iter):
            pbar.close()
            if (iter >= max_iter):
                print("Maximum number of iterations reached.")
            break

    # ========== ADMM convergence summary ==========
    if len(global_sweep_history) > 0:
        g_arr = numpy.array(global_sweep_history)
        print(f"global_admm: {len(g_arr)} calls, sweeps used avg={g_arr.mean():.1f} "
              f"max={g_arr.max()} (cap={inner_maxiter}); hit cap in {global_hit_cap}/{len(g_arr)} "
              f"calls ({100 * global_hit_cap / len(g_arr):.1f}%) "
              f"{global_cg_nonconverged} inner CG solve(s) did not reach atol within cg_maxiter={cg_maxiter}")
        
    return Xori, M + centre