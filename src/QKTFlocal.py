import cupy as np
import numpy
from tqdm import tqdm
from cupyx.scipy.linalg import khatri_rao
from cupyx.scipy.sparse import linalg, eye, csr_matrix

def cov_matern(d, loghyper, x):
    """
    Computes the Matern covariance matrix for a given dimension.

    Args:


    Returns:
    
    """
    ell = np.exp(loghyper[0])
    sf2 = np.exp(2*loghyper[1])
    def f(t):
        if d == 1: return 1
        if d == 3: return 1 + t
        if d == 5: return 1 + t*(1 + t/3)
        if d == 7: return 1 + t*(1 + t*(6 + t)/15)
    def m(t):
        return f(t)*np.exp(-t)
    dist_sq = ((x[:, None] - x[None, :])/ell)**2
    return sf2*m(np.sqrt(d*dist_sq))

def bohman(loghyper, x):
    range_ = np.exp(loghyper[0])
    dis = np.abs(x[:, None] - x[None, :])
    r = np.minimum(dis/range_, 1)
    k = (1 - r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
    k[k < 1e-16] = 0
    k[np.isnan(k)] = 0
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
    return xi - np.maximum((tau - 1)/alpha, np.minimum(xi, tau/alpha)) # applies the proximal operator to the input vector.

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

def local_operator(vec, pos_obs, Kr, gamma, lambda_, N):
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
    x = np.zeros(int(numpy.prod(N))) # zero-pads to vector of length N.
    x[pos_obs] = vec # slices the vector to the length of osberved entires - |Omega|.
    Ap = kronecker_mvm(Kr, x, N) # constructs the covariance matrices via Kronecker MVM.
    return lambda_ * Ap[pos_obs] + gamma * vec

def local_admm(lambda_, gamma, priorvalue, a, v, Kr, pos_obs, total_data, YR_tilde, max_iter, tau):
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
    a_obs = (a.ravel(order = 'F'))[pos_obs[0]] # slices the auxiliary variable to only observed entries.
    v_obs = (v.ravel(order = 'F'))[pos_obs[0]] # slices the Lagrangian multiplier to only observed entries.
    x0 = priorvalue.copy() # sets the initial guess for the ADMM algorithm as the previous iteration of the latent matrix.

    # ========== ADMM iterations =========
    for j in range(max_iter):
        a_prev = a_obs.copy() # stores the previous auxiliary varaible used in convergence checks.
        b = lambda_ * (Y_obs - a_obs - v_obs) # right hand side operator used in Conjugate Gradient method.

        # r-update
        def matvec(v): # performs y = Ax for the linear operator used in the Conjugate Gradient method.
            return local_operator(v, pos_obs, Kr, gamma, lambda_, N) # returns the linear operator used in the Conjugate Gradient method.
        
        ar = linalg.LinearOperator((n_obs, n_obs), matvec=matvec, dtype=b.dtype) # constructs the Linear Operator.
        w, info = linalg.cg(ar, b, x0=x0, atol=1e-4, maxiter=max_iter) # performs the Conjugate Gradient method.

        w_full = np.zeros(total_data)
        w_full[pos_obs] = w
        r = kronecker_mvm(Kr, w_full, N) # calculates r.

        # auxiliary variable update
        zeta = Y_obs - v_obs - r[pos_obs] # calculates zeta = y - x - O_1*r.
        alpha = n_obs * lambda_ # calcualtes the parameter for the Proximal operator.
        a_obs = prox_map(zeta, alpha, tau) # Proximal operator used for the quantile function.
    
        # Lagrangian multiplier update
        v_obs = v_obs + lambda_ * (r[pos_obs] + a_obs - Y_obs) # update for Lagrangian multiplier.

        # convergence criterion.
        res_pri = r[pos_obs] + a_obs - Y_obs
        res_dual = lambda_ * (a_obs - a_prev)
        eps_pri = np.sqrt(n_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(r[pos_obs]), np.linalg.norm(a_obs)), np.linalg.norm(Y_obs))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm(v_obs)

        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break

    return r.ravel(order='F'), a_obs, v_obs

def qktf(I, Omega, lengthscaleR: list, varianceR: list, tapering_range, d_MaternR, R, gamma, lambda_, tau, max_iter, epsilon):
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
    pos_miss = np.where(Omega == 0) # creates a tuple of arrays containing the indices of the missing entries in the tensor - can be used directly for indexing and can be unpacked correctly.
    num_obs = int(numpy.sum(Omega)) # calculates the number of observed entries in the tensor.
    total_data = int(numpy.prod(N)) # calculates the total number of entries in the tensor.

    # Mask construction
    mask_matrix = [unfold(Omega, d) for d in range(D)] # creates a list of D matrices, where each matrix is the mode-d unfolding of Omega.
    mask_matrixT = [mask_matrix[d].T for d in range(D)] # creates a list of D matrices, where each matrix is the transpose of the mode-d unfolding of Omega.
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)] # creates a list of D vectors, where each vector is the flattened version of the mode-d unfolding of Omega.
    pos_obs = [np.where(mask_flat[d] == 1) for d in range(D)] # creates a list of D arrays, containing arrays of observed entries.

    # Data centering
    idx = np.sum(mask_matrix[D-1], axis = 0) > 0 # creates a Boolean mask identifying which columns have at least one observed entry.
    train_matrix = I * Omega # creates a mask of the tensor - setting indices to zero where there is data missing.
    train_matrix = train_matrix[train_matrix > 0] # creates a matrix of only the observed entries into a 1D array.
    Isubmean = I - np.mean(train_matrix) # centers the data by subtracting the mean of the observed entries from all entries in the tensor.
    T = Isubmean * Omega # creates a tensor of the centered observed entries - setting indices to zero where there is data missing.

    # ========== Building covariance matrices ==========
    hyper_Kr = [None] * D # creates an empty list to store the hyperparameters for the global and local covariance tapering, list length = D.
    Kr = [None] * D # creates an empty list to store the covariance matrices for the global and local covariance tapering, list length = D.
    inv_Ku = [None] * D # creates an empty list to store the inverse covariance matrices for the global covariance tapering, list length = D.

    for d in range(D-1): # iterates through each dimension of the input tensor.
        x = np.arange(1, N[d] + 1) # creates a vector of integers from 1 to the size of the current dimension - used as input for the covariance function.

        hyper_Kr[d] = [np.log(lengthscaleR[d]), np.log(varianceR[d]), np.log(tapering_range)] # sets the dth dimension of hyperparameters as log of lengthscale, variance, and tapering range
        TaperM = bohman([hyper_Kr[d][2]], x) # sets the taper as the bohman taper - calls the tapering range.
        Kr[d] = csr_matrix(cov_matern(d_MaternR, hyper_Kr[d][:2], x) * TaperM) # sets the local covariance as sparse matrix - efficient for matrix-vector product.

    Kr[D-1] = np.eye(N[D-1]) # initialises the local covariance in the last dimension as identity matrix.

    # ========== Initialisation for ADMM iterations ==========
    X = T # sets the initial value of the fixed tensor as the centered observed entries.
    X[pos_miss] = T.sum() / num_obs # sets the missing entries of the fixed tensor as the mean of the observed entries.

    Rtensor = np.zeros(N) # initialises the local tensor with the same shape as the input data, filled with zeros.
    a = np.zeros(N) # intialises the auxiliary variable used in the local ADMM algorithm.
    v = np.zeros(N) # initialises the Lagrangian multiplier in the local ADMM algorithm.
    Rvector = Rtensor.ravel(order = 'F') # vectorised local tensor.
    Rvector_temp = Rtensor.ravel(order = 'F') # creates a copy of the vectorised local tensor for use in the ADMM iterations.
    X[pos_miss] = Rtensor[pos_miss] # sets the missing entries of X to the sum of the missing entries of global and local components.

    d_all = np.arange(D) # creates a vector of integers from 0 to D-1 - used for indexing.
    train_norm = np.linalg.norm(T) # calculates the norm of the tensor of the centered observed entries - used for convergence checking.
    last_ten = T.copy() # initialises a tensor to store the value of the fixed tensor from the previous iteration for convergence checking.
    pbar = tqdm(total=max_iter, desc="QKTF Iterations") # creates a progress bar for the ADMM iterations.
    iter = 0 # initialises the iteration counter for the ADMM algorithm.

    while True: # runs the ADMM iterations until the maximum number of iterations is reached.
        Ltensor = X # intialises the local tensor as the residual.
        Ltensor_mask = Ltensor * Omega # masks the local tensor.

        # Actual Local ADMM optimisation call.
        Rvector, a_obs, v_obs = local_admm(lambda_, gamma, Rvector[pos_obs[0]], a, v, Kr, pos_obs[0], total_data, Ltensor_mask, 100, tau)
        Rtensor = Rvector.reshape(N, order = 'F') # reshapes the Rvector back to tensor size.
        Rtensor_unfold = unfold(Rtensor, D-1) # unfolds along the last dimension - used in covariance iteration.
        Rtensor_unfold_obs = Rtensor_unfold[:, idx] # ensures only calculating where there's observed entries in columns.
        Kr[D-1] = np.cov(Rtensor_unfold_obs) # calculates the new covariance in the last dimension.

        X[pos_miss] = Rtensor[pos_miss] # updates the missing entries of the fixed tensor as the sum of the global component and the local tensor.
        Xori = X + np.mean(train_matrix) # adds the mean back into X.

        # Convergence checks
        iter += 1 # increments the iteration counter.
        tol = np.linalg.norm((X - last_ten)) / train_norm # calculates the convergence metric as the relative change in the fixed tensor.
        last_ten = X.copy() # updates the tensor for convergence checking to the current fixed tensor.

        pbar.update(1)
        
        if np.isnan(tol) or np.isinf(tol): # checks for numerical issues in convergence metric.
            pbar.set_postfix({'tol': f'{tol:.2e}', 'epoch': iter})
            break
        
        if (tol < epsilon) or (iter >= max_iter):
            pbar.close()
            if (iter >= max_iter):
                print("Maximum number of iterations reached.")
            break

        
    return Xori, Rtensor