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

def enforce_psd(K, rel_floor=1e-4, abs_floor=1e-4):
    """
    Projects a symmetric matrix onto the nearest PSD matrix with a guaranteed minimum eigenvalue
    
    Args:
        K (n, n): symmetric matrix - empirically re-estimated K_r.
        rel_floor: minimum eigenvalue as a fraction of the largest eigenvalue.
        
    Returns:
        K_psd: PSD matrix with eigenvalues >= rel_floor * max_eig
        """
    K_sym = (K + K.T) / 2 # forces exact symmetry.
    eigvals, eigvecs = cp.linalg.eigh(K_sym) # eigendecompose.
    max_eig = cp.abs(eigvals).max()
    min_eig = cp.abs(eigvals).min()
    floor = cp.maximum(abs_floor, rel_floor * max_eig)
    n_floor_clipped = int((eigvals < floor).sum())
    eigvals_clipped = cp.maximum(eigvals, floor)
    K_psd = eigvecs @ cp.diag(eigvals_clipped) @ eigvecs.T # reconstructs from the clipped spectrum.

    return K_psd, n_floor_clipped

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

def kronecker_mvm(Kr, vec, shape):
    """
    Calculates the local covariance matrices thorugh efficient Kronecker matrix-vector products - tranpose in identity happens implicitly through unfolding and folding.

    Args:
        Kr (list): list of covariance matrices for local operator.
        vec (ndarray): input vector.
        shape (tuple): shape of original tensor.
    
    Returns:
        ndarray: covariances matrices for local operator.
    """
    D = len(shape) # gets the number of dimensions.
    x = vec.reshape(shape, order = 'F') # reshapes to tensor size.

    for d in range(D-1, -1, -1): # iterates through the dimensions.
        x_unfold = unfold(x, d) # mode-d unfolding.
        x_unfold = Kr[d] @ x_unfold # matrix multiplication.
        x = fold(x_unfold, shape, d) # folds back to tensor.

    return x.ravel(order = 'F')

def local_operator(vec, pos_obs, Kr, gamma, lambda_, N, total_data, buffer=None):
    """
    Constructs the local linear operator used in the local ADMM algorithm.

    Args:
        vec (ndarray): vector to be multiplied by the local linear operator.
        pos_obs (list): list of observed entries.
        Kr (list): list of covariance matrices for the local linear operator.
        gamma (float): covariance regularisation parameter.
        lambda_ (float): local ADMM penalty parameter.
        N (int): number of total entries.

    Returns:
        ndarray: linear operator used in the Conjugate Gradient method for the local ADMM optimisation steps of the QKTF algorithm. 
    """
    buffer[:] = 0.0
    buffer[pos_obs] = vec # slices the vector to the length of osberved entires - |Omega|.
    Ap = kronecker_mvm(Kr, buffer, N) # constructs the covariance matrices via Kronecker MVM.
    return lambda_ * Ap[pos_obs] + ((gamma / total_data) * vec)

def local_admm(lambda_, gamma, w_prior, xi, x, Kr, pos_obs, total_data, YR_tilde,
               inner_maxiter, tau, cg_maxiter, rel_tol, abs_tol, verbose=False):
    """
    Local ADMM algorithm for updating the local tensor in the QKTF algorithm.

    Args:
        lambda_ (float): local ADMM penalty parameter.
        gamma (float): covariance regularisation parameter.
        priorvalue (ndarray): previous iteration of local tensor - used as a warm start.
        a (ndarray): auxiliary variable.
        v (ndarray): Lagrangian multiplier.
        Kr (list): list of covariance matrices for local component.
        pos_obs (ndarray): ndarray containing positions where data is not missing.
        total_data (int): number of total data entries.

    Returns:
        r_obs (ndarray): updated local tensor after local ADMM algorithm of QKTF algorithm.
        a_vec (ndarray): updated auxiliary variable after local ADMM algorithm of QKTF algorithm.
        v_vec (ndarray): updated Lagrangian multiplier after local ADMM algorithm of QKTF algorithm.
        info: CG convergence info.
    """
    N = numpy.array(YR_tilde.shape) # gets the shape of YR_tilde and set N to it.
    n_obs = pos_obs[0].shape[0] # collects the number of observations.
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs[0]] # slices the fixed tensor to only observed entries.
    x0 = w_prior.copy() # sets the initial guess for the ADMM algorithm as the previous iteration of the latent matrix.

    w_full = cp.zeros(total_data, dtype=Y_obs.dtype) # zero-padding reused across every CG matvec call.

    def matvec(vec): # performs y = Ax for the linear operator used in the Conjugate Gradient method.
        return local_operator(vec, pos_obs, Kr, gamma, lambda_, N, total_data, buffer=w_full) # returns the linear operator used in the Conjugate Gradient method.

    ar = linalg.LinearOperator((n_obs, n_obs), matvec=matvec, dtype=Y_obs.dtype) # constructs the Linear Operator.

    assert inner_maxiter > 0, "local_admm requires at least one ADMM sweep"

    # ========== ADMM iterations =========
    converged = False
    lcg_nonconverged = 0
    for j in range(inner_maxiter):
        x_prev = x.copy() # stores the previous auxiliary varaible used in convergence checks.
        w_prev = x0.copy() # stores the previous local tensor used in convergence checks.
        b = lambda_ * (Y_obs - x) + xi # right hand side operator used in Conjugate Gradient method.

        if verbose:
            print(f"[local_admm]sweep {j}: a norm={cp.linalg.norm(x):.2e}, v norm={cp.linalg.norm(xi):.2e}, b norm={cp.linalg.norm(b):.2e}")
        
        w, info = linalg.cg(ar, b, x0=x0, atol=1e-4, maxiter=cg_maxiter) # performs the Conjugate Gradient method.

        if info != 0:
            lcg_nonconverged += 1
        if verbose: 
            print(f"[local_admm] sweep {j}: CG info={info}")

        x0 = w

        if verbose:
            print(f"[local_admm] sweep {j}: x0 norm={cp.linalg.norm(x0):.2e}, w norm = {cp.linalg.norm(w):.2e}")

        w_full[:] = 0.0
        w_full[pos_obs] = w
        r = kronecker_mvm(Kr, w_full, N) # calculates r.

        # auxiliary variable update
        alpha = n_obs * lambda_ # calcualtes the parameter for the Proximal operator.
        x = prox_map(Y_obs - r[pos_obs] + (xi/lambda_), alpha, tau) # Proximal operator used for the quantile function.
    
        # Lagrangian multiplier update
        xi = xi - lambda_ * (r[pos_obs] + x - Y_obs)

        if verbose:
            print(f"[local_admm] sweep {j}: a norm={cp.linalg.norm(x):.2e}, v norm={cp.linalg.norm(xi):.2e}, r norm={cp.linalg.norm(r):.2e}")

        # convergence criterion.
        res_pri = r[pos_obs] + x - Y_obs
        res_dual = lambda_ * (x - x_prev)
        eps_pri = cp.sqrt(n_obs) * abs_tol + rel_tol * cp.maximum(cp.maximum(cp.linalg.norm(r[pos_obs]), cp.linalg.norm(x)), cp.linalg.norm(Y_obs))
        eps_dual = cp.sqrt(total_data) * abs_tol + rel_tol * cp.linalg.norm(xi)
        
        if cp.linalg.norm(res_pri) <= eps_pri and cp.linalg.norm(res_dual) <= eps_dual:
            converged = True
            break

    return r.ravel(order='F'), w, x, xi, j+1, converged, lcg_nonconverged

def QKTFlocal(I, Omega, lengthscaleR: list, varianceR: list,
        tapering_range, d_MaternR, R, gamma, lambda_, tau, max_iter,
        epsilon, inner_maxiter, cg_maxiter=500, verbose=False):
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
    assert len(lengthscaleR) == D # ensures the number of lengthscales providede matches the number of dimensions of the input tensor.
    assert len(varianceR) == D # ensures the number of variances provided matches the number of dimensions of the input tensor.
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
    hyper_Kr = [None] * D # creates an empty list to store the hyperparameters for the global and local covariance tapering, list length = D.
    Kr = [None] * D # creates an empty list to store the covariance matrices for the global and local covariance tapering, list length = D.

    for d in range(D-1): # iterates through each dimension of the input tensor.
        a = cp.arange(1, N[d] + 1) # creates a vector of integers from 1 to the size of the current dimension - used as input for the covariance function.

        hyper_Kr[d] = [cp.log(lengthscaleR[d]), cp.log(varianceR[d]), cp.log(tapering_range)] # sets the dth dimension of hyperparameters as log of lengthscale, variance, and tapering range
        TaperM = bohman([hyper_Kr[d][2]], a) # sets the taper as the bohman taper - calls the tapering range.
        Kr[d] = csr_matrix(cov_matern(d_MaternR, hyper_Kr[d][:2], a) * TaperM) # sets the local covariance as sparse matrix - efficient for matrix-vector product.
    Kr[D-1] = csr_matrix(eye(N[D-1])) # initialises the local covariance in the last dimension as identity matrix.

    # ========== Initialisation for ADMM iterations ==========
    X = T.copy() # sets the initial value of the fixed tensor as the centered observed entries.
    X[pos_miss] = T.sum() / num_obs # sets the missing entries of the fixed tensor as the mean of the observed entries.

    Rtensor = cp.zeros(N) # initialises the local tensor with the same shape as the input data, filled with zeros.
    w_warm = cp.zeros(num_obs)
    x = cp.zeros(num_obs) # intialises the auxiliary variable used in the local ADMM algorithm.
    xi = cp.zeros(num_obs) # initialises the Lagrangian multiplier in the local ADMM algorithm.
    Rvector = Rtensor.ravel(order = 'F') # vectorised local tensor.
    Rvector_temp = Rtensor.ravel(order = 'F') # creates a copy of the vectorised local tensor for use in the ADMM iterations.
    X[pos_miss] = Rtensor[pos_miss] # sets the missing entries of X to the sum of the missing entries of global and local components.

    d_all = cp.arange(D) # creates a vector of integers from 0 to D-1 - used for indexing.
    train_norm = cp.linalg.norm(T) # calculates the norm of the tensor of the centered observed entries - used for convergence checking.
    last_ten = T.copy() # initialises a tensor to store the value of the fixed tensor from the previous iteration for convergence checking.
    pbar = tqdm(total=max_iter, desc="QKTFlocal Iterations") # creates a progress bar for the ADMM iterations.
    iter = 0 # initialises the iteration counter for the ADMM algorithm.
    local_sweep_history = []
    local_hit_cap = 0
    local_cg_nonconverged = 0
    kr_last_history = []

    while True: # runs the ADMM iterations until the maximum number of iterations is reached.
        l_sweeps, l_converged = None, None
    
        Ltensor = X
        Ltensor_mask = Ltensor * Omega # masks the local tensor.

           # Actual Local ADMM optimisation call.
        Rvector, w_warm, x, xi, l_sweeps, l_converged, lcg_nonconverged = local_admm(
            lambda_, gamma, w_warm, xi, x, Kr, pos_obs[0], total_data,
            Ltensor_mask, inner_maxiter, tau, cg_maxiter=cg_maxiter, rel_tol=1e-4, abs_tol=1e-4, verbose=verbose
            )

        local_sweep_history.append(l_sweeps)
        if not l_converged:
            local_hit_cap += 1
        local_cg_nonconverged += lcg_nonconverged

        Rtensor = Rvector.reshape(N, order = 'F') # reshapes the Rvector back to tensor size.
        Rtensor_unfold = unfold(Rtensor, D-1) # unfolds along the last dimension - used in covariance iteration.
        Rtensor_unfold_obs = Rtensor_unfold[:, idx] # ensures only calculating where there's observed entries in columns.
        Kr_raw = cp.cov(Rtensor_unfold_obs) # calculates the new covariance in the last dimension.
        Kr_psd, n_clip = enforce_psd(Kr_raw)
        a_last = cp.arange(1, N[D-1] + 1)
        TaperM_last = bohman([cp.log(tapering_range)], a_last)
        Kr_tapered = Kr_psd * TaperM_last
        Kr[D-1] = csr_matrix(Kr_tapered)

        kr_raw_max = float(cp.abs(Kr_raw).max())
        kr_tapered_max = float(cp.abs(Kr_tapered).max())
        kr_psd_max = float(Kr[D-1].max())
        kr_last_history.append((kr_raw_max, kr_tapered_max, kr_psd_max))
        if verbose:
            print(f"[qktf] iter {iter}: Kr[D-1] max raw={kr_raw_max:.4g} "
                  f"tapered={kr_tapered_max:.4g} post-PSD={kr_psd_max:.4g}")

        X[pos_miss] = Rtensor[pos_miss] # updates the missing entries of the fixed tensor as the sum of the global component and the local tensor.
        Xori = X + centre # adds the mean back into X.

        print(f"[QKTFlocal] Rtensor_norm={cp.linalg.norm(Rtensor)}",
              f"[QKTFlocal] Xori_norm={cp.linalg.norm(Xori)}")

        # Convergence checks
        iter += 1 # increments the iteration counter.
        tol = cp.linalg.norm((X - last_ten)) / train_norm # calculates the convergence metric as the relative change in the fixed tensor.
        last_ten = X.copy() # updates the tensor for convergence checking to the current fixed tensor.

        pbar.update(1)
        pbar.set_postfix({
            'tol': f'{tol:.2e}',
            'l_sweeps': ('-' if l_sweeps is None else l_sweeps)
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
    if len(local_sweep_history) > 0:
        l_arr = numpy.array(local_sweep_history)
        print(f"local_admm: {len(l_arr)} calls, sweeps used avg={l_arr.mean():.1f} "
              f"max={l_arr.max()} (cap={inner_maxiter}); hit cap in {local_hit_cap}/{len(l_arr)} "
              f"calls ({100 * local_hit_cap / len(l_arr):.1f}%) "
              f"{local_cg_nonconverged} inner CG solve(s) did not reach atol within cg_maxiter={cg_maxiter}")

    if len(kr_last_history) > 0:
        kr_arr = numpy.array(kr_last_history)  # columns: raw_max, tapered_max, psd_max, n_ceil
        print(f"Kr[D-1] magnitude over {len(kr_arr)} re-estimation(s): "
              f"raw max range=[{kr_arr[:, 0].min():.4g}, {kr_arr[:, 0].max():.4g}], "
              f"post-taper range=[{kr_arr[:, 1].min():.4g}, {kr_arr[:, 1].max():.4g}], "
              f"post-PSD range=[{kr_arr[:, 2].min():.4g}, {kr_arr[:, 2].max():.4g}]")
        
    return Xori, Rtensor