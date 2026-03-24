import cupy as np
import numpy
from cupyx.scipy.sparse import linalg, eye, csr_matrix
from cupyx.scipy.linalg import khatri_rao
from tqdm import tqdm

def cov_matern(d, loghyper, x):
    ell = np.exp(loghyper[0])
    sf2 = np.exp(2*loghyper[1])
    def f(t):
        if d == 1: return 1
        if d == 3: return 1+t
        if d == 5: return 1+t*(1+t/3)
        if d == 7: return 1+t*(1+t*(6+t)/15)
    def m(t):
        return f(t)*np.exp(-t)
    dist_sq = ((x[:, None] - x[None, :])/ell)**2
    return sf2*m(np.sqrt(d*dist_sq))

def bohman(loghyper, x):
    range_ = np.exp(loghyper)
    dis = np.abs(x[:, None] - x[None, :])
    r = np.minimum(dis/range_, 1)
    k = (1-r)*np.cos(np.pi*r)+np.sin(np.pi*r)/np.pi
    k[k < 1e-16] = 0
    k[np.isnan(k)] = 0 
    return k

def unfold(tensor, mode):
    """
    Performs mode-d unfolding of a tensor.

    Args:
        tensor: input tensor
        mode: axis to do unfolding

    Returns:
        np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F'): unfolded tensor
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def fold(mat, dim, mode):
    """
    Performs folding of an unfolded tensor.

    Args:
        mat: unfolded tensor
        dim: tuple/1d array containing dimensions of original tensor
        mode: axis where tensor was unfolded

    Returns:
        np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode): folded tensor
    """
    index = list()
    index.append(mode) 
    for i in range(dim.shape[0]): 
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

def build_khatri_rao(U, dims):
    """
    Builds khatri-rao product for arbitrary dimension.
    
    Args:
        U: list of latent matrices in each dimension
        d: number of dimensions
    
    Returns:
        khatri-rao product
    """
    dims = [int(d) for d in dims]
    if len(dims) == 1:
        return U[dims[0]]
    else:
        result = U[dims[-1]]
        for i in range(len(dims) - 2, -1, -1):
            result = khatri_rao(result, U[dims[i]])
        return result
    
def reconstruct_tensor(U, shape):
    """
    Reconstructs tensor from CP for arbitrary dimension (used in global).

    Args:
        U: list of latent matrices
        dims: number of dimensions
        R: rank of CP decomposition

    Returns:
        M: reconstructed tensor 
    """
    D = len(shape)
    dims_except_0 = list(range(1, D))
    if len(dims_except_0) > 0:
        KrU = build_khatri_rao(U, dims_except_0)
        M_unfold = U[0] @KrU.T
    else:
        M_unfold
    
    M = M_unfold.reshape(shape, order = 'F')

    return M

def kronecker_covariances(Kr, vec, shape):
    """
    Creates kronecker product of covariances for local component.

    Args:
        Kr: list of covariance matrices
        vec: input vector
        shape: shape of original tensor

    Returns:
        result: kronecker product of covariances
    """
    D = len(shape)
    x = vec.reshape(shape, order = 'F')

    for d in range(D):
        x_unfold = unfold(x, d)
        x_unfold = Kr[d] @ x_unfold
        x = fold(x_unfold, numpy.array(shape), d)
    
    return x.ravel(order = 'F')

def prox_map(xi, alpha, tau):
        """
        Proximal mapping for quantile regression.

        Args:
            xi: input vector
            alpha: sum_obs * ADMM penalty
            tau: quantile parameter
        """
        low = (tau - 1) / alpha
        high = tau / alpha
        return xi - np.maximum((tau - 1) / alpha, np.minimum(xi, tau / alpha))    

def global_operator(vec, maskT, KrU, KrU_T, Qu, psi, sigma, R, M):
    """
    Constructs linear operator used in global ADMM algorithm.

    Args:
        vec: input vector
        maskT: mask matrix transposed
        KrU: khatri-rao product for construction of H_d
        KrU_T: transpose of KrU
        Qu: covariance matrix for global component in dimension d
        psi: global regularisation
        sigma: global ADMM penalty
        R, M: dimensions of input tensor in dimension d
    """
    X = vec.reshape(R, M, order = 'F')
    temp = KrU @ X
    temp *= maskT
    Ap1 = sigma * (KrU_T @ temp)
    Ap2 = psi * (X @ Qu)
    return (Ap1 + Ap2).ravel(order = 'F')

def global_admm(Qu, psi, sigma, KrU, mask_matrixT, YR_tilde, priorvalue, max_iter, tau, z, theta, sum_obs, total_data, R):
    """
    Global ADMM algorithm for updating latent matrices in each dimension.

    Args:
        Qu: covariance matrix for global component in dimension d
        psi: global regularisation
        sigma: global ADMM penalty
        KrU: khatri-rao product for construction of H_d
        mask_matrixT: mask matrix transposed
        YR_tilde: input tensor
        priorvalue: initial value for latent matrix in dimension d
        max_iter: maximum number of iterations
        tau: quantile parameter
        z: initial value for auxiliary variable
        theta: initial value for lagrangian multiplier
        sum_obs: number of observed entries
        total_data: total number of data entries
    
    Returns:
        u_vec: updated latent matrix in dimension d
        z_vec: updated auxiliary variable
        theta_flat: updated lagrangian multiplier
        info: CG convergence information
    """
    M = YR_tilde.shape[1]
    x0 = priorvalue.copy()
    KrU_T = KrU.T
    n = R * M

    # ========== ADMM iterations ==========
    for j in range(max_iter):
        z_prev = z.copy()
        rhs_mat = sigma * (YR_tilde - z) - theta
        rhs_mat *= mask_matrixT
        b_mat = KrU_T @ rhs_mat
        b = b_mat.ravel(order = 'F')
    
        # latent matrix update
        def matvec(v):
            return global_operator(v, mask_matrixT, KrU, KrU_T, Qu, psi, sigma, R, M)
        A_op = linalg.LinearOperator((n, n), matvec=matvec, dtype=b.dtype)
        u_vec, gcg_info = linalg.cg(A_op, b, x0 = x0, atol = 1e-4, maxiter = max_iter)
    
        # auxiliary variable update
        u_mat = u_vec.reshape(R, M, order = 'F')
        H_u = KrU @ u_mat
        H_u += mask_matrixT
        alpha = sum_obs * sigma
        eta = YR_tilde - theta - H_u

        z = prox_map(eta, alpha, tau)
    
        # lagrangian multiplier update
        theta = theta + H_u + z - YR_tilde

        # convergence criterion
        res_pri = H_u + z - (mask_matrixT * YR_tilde)
        res_dual = sigma * KrU_T @ (mask_matrixT * (z - z_prev))
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(H_u), np.linalg.norm(z)), np.linalg.norm(YR_tilde))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm(KrU_T @ theta)

        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break  

    return u_vec, z, theta, gcg_info
    
def local_operator(vec, pos_obs, Kr, lambda_, gamma, shape):
    """
    Constructs linear operator for local ADMM algorithm.

    Args:
        vec: input vector
        pos_obs: indices of observed entries
        Kd, Kt, Ks: covariance matrices for local component in each dimension
        lambda_: local ADMM regularisation
        d1, d2, d3: dimensions of tensor
        
    Returns:
        Ap[pos_obs] + lambda_ * vec: linear operator
    """
    N = int(numpy.prod(shape))
    x = np.zeros(N)
    x[pos_obs] = vec
    Ap = gamma * kronecker_covariances(Kr, x, shape)
    return Ap[pos_obs] + lambda_ * vec

def local_admm(lambda_, gamma, priorvalue, a, v, Kr, pos_obs, sum_obs, YR_tilde, max_iter, tau, total_data, num_obs):
    """
    Local ADMM algorithm

    Args:
        lambda_: local ADMM regularisation
        priorvalue: initial value for rtensor
        a: initial value for auxiliary variable
        v: initial value for lagrangian multiplier
        Kd, Kt, Ks: covariance matrix for local component in each dimension
        pos_obs: indices of observed entries
        sum_obs: number of observed entries
        YR_tilde: input tensor
        mask_matrixT: mask matrix transposed
        max_iter: maximum number of iterations 
        tau: quantile parameter
        total_data: total number of data entries

    Returns:
        r_obs = updated residual for observed entries
        a_vec = updated auxiliary variable
        v_vec = updated lagrangian multiplier
        info = CG convergence information
    """
    shape = YR_tilde.shape
    shape_arr = numpy.array(shape)
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs]
    r = priorvalue.copy()
    a_vec = a.ravel(order = 'F')[pos_obs]
    v_vec = v.ravel(order = 'F')[pos_obs]
    N = int(numpy.prod(shape_arr))

    # ========== ADMM iterations ==========
    for j in range(max_iter):
        a_prev = a_vec.copy()
        rhs_mat = Y_obs - a_vec - v_vec
        b = np.zeros(num_obs)
        b = lambda_ * rhs_mat

        # r-update
        ar = linalg.LinearOperator((num_obs, num_obs), matvec=lambda v: local_operator(v, pos_obs, Kr, lambda_, gamma, shape))
        r_vec, lcg_info = linalg.cg(ar, b, x0 = np.zeros(num_obs), atol = 1e-4, maxiter = max_iter)

        # auxiliary variable update
        r_obs = r_vec
        zeta = Y_obs - v_vec - r_obs
        alpha = sum_obs * lambda_

        a_vec = prox_map(zeta, alpha, tau)

        # lagrangian multiplier update
        v_vec = v_vec + r_obs + a_vec - Y_obs

        # convergence criterion
        res_pri = r_obs + a_vec - Y_obs
        res_dual = lambda_ * (a_vec - a_prev)
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(r_obs), np.linalg.norm(a_vec)), np.linalg.norm(Y_obs))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm(v_vec)

        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break

    return r_obs, a_vec, v_vec, lcg_info

def qktf(I, Omega, lengthscaleU: list, lengthscaleR: list, varianceU: list, varianceR: list, tapering_range, d_maternU, d_maternR, R, psi, sigma, lambda_, gamma, tau, maxiter, K0, epsilon):
    """
    QKTF for an arbitrary tensor.
    
    Args:
        I: Input tensor (arbitrary dimension)
        Omega: Binary mask (same shape as I)
        lengthscaleU: List of lengthscales for each dimension (reference for comparing dimensions)
        lengthscaleR: List of lengthscales for each dimension - local component (reference for comparing dimensions)
        varianceU: List of variances for each dimension - global
        varianceR: List of variances for each dimension - local
        tapering_range: Tapering range for local component
        d_maternU: Degree of Matern kernel for global
        d_maternR: Degree of Matern kernel for local
        R: Tensor rank - pre-specified through CP decomposition
        psi: regularisation for global component
        sigma: global ADMM penalty
        lambda_: local ADMM penalty
        tau: quantile parameter
        maxiter: maximum number of iterations
        K0: number of iterations before local component is updated
        epsilon: convergence criterion

    Returns:
        I_recovery: recovered tensor
        R_component: local component of recovery
        M_component: global component of recovery
    """
    # ========== Setup ==========
    N = I.shape
    N = numpy.array(N)
    D = len(N)
    total_data = numpy.prod(N)
    maxP = float(np.max(I))

    # Validate inputs
    assert len(lengthscaleU) == D
    assert len(lengthscaleR) == D
    assert len(varianceU) == D
    assert len(varianceR) == D
    assert I.shape == Omega.shape
    assert R > 0
    assert 0 < tau < 1

    # ========== Pre-processing ==========

    # Binary indicator matrix
    Omega = Omega.astype(bool)
    pos_miss = np.where(Omega == 0)
    num_obs = int(np.sum(Omega))

    # Mask construction
    mask_matrix = [unfold(Omega, d) for d in range(D)]
    mask_matrixT = [mask_matrix[d].T for d in range(D)]
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)]
    pos_obs = [np.where(mask_flat[d] == 1)[0] for d in range(D)]
    pos_full = np.where(Omega.ravel(order = 'F') == 1)[0]

    # Data centering
    train_matrix = I * Omega
    train_matrix = train_matrix[train_matrix > 0]
    Isubmean = I - np.mean(train_matrix)
    T = Isubmean * Omega

    # ========== Build covariance matrices ==========
    hyper_Ku, hyper_Kr = [None] * D, [None] * D
    Ku, Kr = [None] * D, [None] * D
    invKu = [None] * D

    for d in range(D):
        x = np.arange(1, N[d] + 1)

        # Global covariance
        hyper_Ku[d] = [np.log(lengthscaleU[d]), np.log(varianceU[d])]
        Ku[d] = cov_matern(d_maternU, hyper_Ku[d], x)
        invKu[d] = np.linalg.inv(Ku[d])

        # Local covariance
        hyper_Kr[d] = [np.log(lengthscaleR[d]), np.log(varianceR[d]), np.log(tapering_range)]
        TaperM = bohman(hyper_Kr[d][0], x)
        Kr[d] = csr_matrix(cov_matern(d_maternR, hyper_Kr[d][:2], x) * TaperM)

    # ========== Initialisation for ADMM iterations ==========
    X = T
    X[pos_miss] = T.sum() / num_obs

    # Variable initialisation
    z = [np.random.randn(N[d], R) * 0.1 for d in range(D)]
    theta = [np.random.randn(N[d], R) * 0.1 for d in range(D)]
    U = [np.random.randn(N[d], R) * 0.1 for d in range(D)] 

    rtensor = np.zeros(N) 
    y = np.zeros(N) 
    v = np.zeros(num_obs) 
    a = np.zeros(num_obs)

    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rvector = rtensor.ravel(order='F')
    rvector_temp = rtensor.ravel(order = 'F')

    # Initial reconstruction
    M = reconstruct_tensor(U, N)
    X[pos_miss] = M[pos_miss] + rtensor[pos_miss]

    d_all = np.arange(0, D) # array of all dimensions
    train_norm = np.linalg.norm(T)
    last_ten = T.copy()
    pbar = tqdm(total=maxiter, desc="QKTF Iterations")
    iter = 0

    # ========== Main algorithm iterations ==========

    while iter < maxiter:

        # Update global component
        Gtensor = X - rtensor 
        Gtensor_mask = Gtensor * Omega

        for d in range(D):
            dsub = np.delete(d_all, d)
            dsub = np.array(dsub)
            KrU = build_khatri_rao(U, dsub)
            Gtensor_unfold = unfold(Gtensor_mask, d).T

            UTvector[d], z[d], theta[d], gcg_info = global_admm(
                invKu[d], psi, sigma, KrU, mask_matrixT[d], Gtensor_unfold, UTvector[d], 1000, tau, z[d], theta[d], num_obs, total_data, R)
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T
        
        # Reconstruct global component
        M = reconstruct_tensor(U, N)    
    
        X[pos_miss] = M[pos_miss] + rtensor[pos_miss]

        # Update local component
        if iter >= K0:
            Ltensor = X - M
            Ltensor_mask = Ltensor * Omega

            rvector_temp, a, v, lcg_info = local_admm(
                lambda_, gamma, rvector_temp[pos_full], a, v, Kr, pos_full, num_obs, Ltensor_mask, 1000, tau, total_data, num_obs)
            
            rvector_full = np.zeros(total_data)
            rvector_full[pos_full] = rvector_temp
            rvector_kc = kronecker_covariances(Kr, rvector_full, N)
            rvector = rvector_kc
            rtensor = rvector.reshape(N, order = 'F')

            # Updating local covariances
            if D >= 2:
                rtensor_unfold_last = unfold(rtensor, D-1)
                idx_last = np.sum(mask_matrix[D-1], axis = 0) > 0
                rtensor_obs = rtensor_unfold_last[:, idx_last]
                Kr[D-1] = np.cov(rtensor_obs)      
            
        else:
            rtensor = np.zeros(N)

        # Update X
        X[pos_miss] = M[pos_miss] + rtensor[pos_miss]

        # Convergence checks
        iter += 1
        tol = np.linalg.norm((X - last_ten)) / train_norm
        last_ten = X.copy()
        pbar.set_postfix({'tol': f'{tol:.2e}', 'M_norm': f'{np.linalg.norm(M):.2e}', 'R_norm': f'{np.linalg.norm(rtensor):.2e}'})
        pbar.update(1)

        if (iter >= maxiter) or (tol < epsilon):
            pbar.close()
            break

        I_recovery = X + np.mean(train_matrix)

    return I_recovery, M + np.mean(train_matrix), rtensor



        





