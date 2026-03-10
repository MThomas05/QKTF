import cupy as np
import numpy
from cupyx.scipy.sparse import linalg, eye, csr_matrix
from cupyx.scipy.linalg import khatri_rao
import verbose

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
    range_ = np.exp(loghyper[0])
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
    index = list() # defines new axis order - encodes which axes became rows
    index.append(mode) # ensures 'mode' axis is first
    for i in range(dim.shape[0]): # appends all other axes in order
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
    if len(dims) == 1:
        return U[dims[0]]
    else:
        result = U[dims[-1]]
        for i in reversed(dims[:-1]):
            return khatri_rao(U[i], result)
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
    dims = len(U)
    N = numpy.array(shape)
    dims_except_0 = np.arange(1, dims)
    KrU = build_khatri_rao(U, dims_except_0)
    M_unfold = U[0] @ KrU.T
    M = fold(M_unfold, N, 0)
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

def global_admm(Qu, psi, sigma, KrU, mask_matrixT, YR_tilde, priorvalue, max_iter, tau, z, theta, sum_obs, total_data):
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
    R, M = YR_tilde.shape
    YR_flat = YR_tilde.ravel(order = 'F')
    x0 = priorvalue.copy()
    x0_flat = x0.ravel(order = 'F')
    z_flat = z.ravel(order = 'F')
    theta_flat = theta.ravel(order = 'F')
    KrU_T = KrU.T

    for j in range(max_iter):
        z_prev = z_flat.copy()
        rhs_mat = mask_matrixT * (YR_tilde - z - theta)
        b_mat = sigma * (KrU_T @ rhs_mat)
        b = b_mat.ravel(order='F')
        #---------- u-update ----------
        A_op = linalg.LinearOperator((R*M), (R*M), matvec=lambda v: global_operator(v, mask_matrixT, KrU, KrU_T, Qu, psi, sigma, R, M))
        u_vec, info = linalg.cg(A_op, b, x0 = x0_flat, atol = 1e-4, maxiter = max_iter)
    
        #---------- z-update (proximal mapping) ----------
        alpha = sum_obs * sigma
        eta = YR_flat - theta_flat - u_vec

        z_vec = prox_map(eta, alpha, tau)
    
        #---------- x-update (lagrangian multiplier) ----------
        theta_flat = theta_flat + u_vec + z_vec - YR_flat

        #---------- residuals ----------
        res_pri = u_vec + z_vec - (mask_matrixT * YR_flat)
        res_dual = sigma * KrU_T @ (mask_matrixT * (z_vec - z_prev))
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(u_vec), np.linalg.norm(z)), np.linalg.norm(YR_tilde))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm((mask_matrixT @ KrU_T) * theta)
        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break    

    return u_vec, z_vec, theta_flat, info
    
def local_operator(vec, pos_obs, Kr, lambda_, shape):
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
    x = np.zeros(shape)
    x[pos_obs] = vec
    Ap = kronecker_covariances(Kr, x, shape)
    return Ap[pos_obs] + lambda_ * vec

def local_admm(lambda_, priorvalue, a, v, Kr, pos_obs, sum_obs, YR_tilde, mask_matrixT, max_iter, tau, total_data):
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
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs]
    r = priorvalue.copy()
    a_vec = a.ravel(order = 'F')
    v_vec = v.ravel(order = 'F')
    N = np.prod(shape)

    #---------- admm ----------
    for j in range(max_iter):
        a_prev = a_vec.copy()
        rhs_mat = Y_obs - a_vec - v_vec
        b = np.zeros(N)
        b[pos_obs] = lambda_ * rhs_mat
        #---------- r-update ----------
        ar = linalg.LinearOperator((N, N), matvec=lambda v: local_operator(v, pos_obs, Kd, Kt, Ks, lambda_, d1, d2, d3))
        r_init = np.zeros(N)
        r_vec, info = linalg.cg(ar, b, x0 = r_init, atol = 1e-4, maxiter = max_iter)

        #---------- z-update ---------
        r_obs = r_vec[pos_obs]
        zeta = Y_obs - v_vec - r_obs
        alpha = sum_obs * lambda_

        a_vec = prox_map(zeta, alpha, tau)

        #---------- x-update ----------
        v_vec = v_vec + r_obs + a_vec - Y_obs

        #---------- convergence criterion ----------
        res_pri = r_obs + a_vec - Y_obs
        res_dual = lambda_ * (a_vec - a_prev)
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(r_obs), np.linalg.norm(a_vec)), np.linalg.norm(Y_obs))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm(v_vec)

        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break

    return r_obs, a_vec, v_vec, info

def qktf(I, Omega, lengthscaleU: list, lengthscaleR: list, varianceU: list, varianceR: list, tapering_range, d_maternU, d_maternR, R, psi, sigma, lambda_, tau, maxiter, K0, epsilon):
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
        Xori: recovered tensor
        rtensor + np.mean(train_matrix): local component of recovery
        M + np.mean(train_matrix): global component of recovery
    """
    # ========== Setup ==========
    N = I.shape
    N = numpy.array(N)
    D = len(N)
    total_data = numpy.prod(N)
    maxP = float(np.max(I))

    # Validate inputs
    assert len(lengthscaleU) == D, f"lengthscaleU must have length {D}, it's got {len{lengthscaleU}}"
    assert len(lengthscaleR) == D, f"lengthscaleR must have length {D}, it's got {len{lengthscaleR}}"
    assert len(varianceU) == D, f"varianceU must have length {D}, it's got {len{varianceU}}"
    assert len(varianceR) == D, f"varianceR must have length {D}, it's got {len{varianceR}}"
    assert I.shape == Omega.shape, f"I and Omega must have the same shape, but got {I.shape} and {Omega.shape}"
    assert R > 0, f"Rank R must be positive, but got {R}"
    assert 0 < tau < 1, f"Quantile parameter tau must be in (0, 1), but got {tau}"

    # ========== Pre-processing ==========

    # Binary indicator matrix
    Omega = Omega.astype(bool)
    pos_miss = np.where(Omega == 0)
    num_obs = int(np.sum(Omega))

    # Mask construction
    mask_matrix = [unfold(Omega, d) for d in range(D)]
    mask_matrixT = [mask_matrix[d].T for d in range(D)]
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)]
    pos_obs = [np.where(mask_flat[d] == 1) for d in range(D)]

    # Data centering
    idx = np.sum(mask_matrix[2], axis = 0) > 0
    train_matrix = I * Omega
    train_matrix = train_matrix[train_matrix > 0]
    Isubmean = I - np.mean(train_matrix)
    T = Isubmean * Omega

    # ========== Build covariance matrices ==========
    hyper_Ku, hyper_Kr = [None] * D, [None] * D
    Ku, Kr = [None] * D, [None] * D
    invKu = [None] * D

    if verbose:
        print("Building covariance matrices...")

    hyper_Ku[0] = [np.log(lengthscaleU[0]), np.log(varianceU[0])]
    hyper_Ku[1] = [np.log(lengthscaleU[1]), np.log(varianceU[1])]
    
    hyper_Kr = [None] * D
    hyper_Kr[0] = [np.log(lengthscaleR[0]), np.log(varianceR[0]), np.log(tapering_range)]
    hyper_Kr[1] = [np.log(lengthscaleR[1]), np.log(varianceR[1]), np.log(tapering_range)]

    Ku, Kr = [None] * D, [None] * D
    invKu = [None] * D

    x = np.arange(1, N[0] + 1)
    Ku[0] = cov_matern(d_maternU, hyper_Ku[0], x)
    invKu[0] = np.linalg.inv(Ku[0])
    TaperM = bohman([hyper_Kr[0][2]], x)
    Kr[0] = csr_matrix(cov_matern(d_maternR, hyper_Kr[0][:2], x) * TaperM)

    x = np.arange(1, N[1] + 1)
    Ku[1] = cov_matern(d_maternU, hyper_Ku[1], x)
    invKu[1] = np.linalg.inv(Ku[1])
    TaperM = bohman([hyper_Kr[1][2]], x)
    Kr[1] = csr_matrix(cov_matern(d_maternR, hyper_Kr[1][:2], x) * TaperM)

    invKu[2] = csr_matrix(eye(N[2]))
    Kr[2] = csr_matrix(eye(N[2]))

    #---------- Initialisations ----------
    X = T
    X[pos_miss] = T.sum() / num_obs
    theta = [np.zeros(int(np.sum(mask_matrix[d]))) for d in range(D)] # Initialise theta as 0
    z = [np.zeros(int(np.sum(mask_matrix[d]))) for d in range(D)] # Initialise z as 0
    U = [np.zeros((R, N[d])) for d in range(D)] # Initialise latent matrices to 0
    rtensor = np.zeros(N) # Initialises r to 0
    y = np.zeros(N) # Initialises y to 0
    v = np.zeros(total_data) # Initialises v to 0
    a = np.zeros(total_data)
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rvector = rtensor.ravel(order='F')
    rvector_temp = rtensor.ravel(order = 'F')
    M_unfold1 = U[0] @ khatri_rao(U[2], U[1]).T ###
    M = fold(M_unfold1, N, 0) ###
    X[pos_miss] = M[pos_miss] + rtensor[pos_miss]
    d_all = np.arange(D) # array of all dimensions
    train_norm = np.linalg.norm(T)
    last_ten = T.copy()
    psnrf = np.zeros(maxiter)
    approxU = [None] * D
    iter = 0

    #-------- Algorithm iterations --------
    while True:
        Gtensor = X - rtensor # G_Omega in latent matrix optimisation
        Gtensor_mask = Gtensor * Omega

        for d in range(D):
            dsub = np.delete(d_all, d)
            dsub = np.array(dsub)
            KrU = khatri_rao(U[dsub[1]], U[dsub[0]])
            HG = KrU.T @ unfold(Gtensor_mask, d).T
            UTvector[d], z[d], theta[d], info = global_admm(invKu[d], psi, sigma, KrU, mask_matrixT[d], HG, UTvector[d], 100, tau, z[d], theta[d], num_obs, total_data)
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T
        M_unfold1 = U[0] @ (khatri_rao(U[2], U[1]).T)
        M = fold(M_unfold1, N, 0)
        X[pos_miss] = M[pos_miss] + rtensor[pos_miss]
        if iter >= K0:
            Ltensor = X - M
            Ltensor_mask = Ltensor * Omega
            rvector_temp[pos_obs[0]], a[pos_obs[0]], v[pos_obs[0]], info = local_admm(lambda_, rvector_temp[pos_obs[0]], a[pos_obs[0]], v[pos_obs[0]], Kr[2], Kr[1], Kr[0], pos_obs[0], num_obs, Ltensor_mask, mask_matrixT, 100, tau, total_data)
            rvector = kronecker_mvm(Kr[2], Kr[1], Kr[0], rvector_temp, N[0], N[1], N[2])
            rtensor = rvector.reshape(N, order = 'F')
            rtensor_unfold3 = unfold(rtensor, 2)
            rtensor_unfold3_obs = rtensor_unfold3[:, idx]
            Kr[2] = np.cov(rtensor_unfold3_obs)
        else:
            rtensor = np.zeros_like(rtensor)
        
        #---------- Diagnostics --------
        X[pos_miss] = M[pos_miss] + rtensor[pos_miss]
        Xori = X + np.mean(train_matrix)
        Xrecovery = np.maximum(0, Xori)
        Xrecovery = np.minimum(maxP, Xrecovery)
        mseC1 = np.linalg.norm(I[:, :, 0].astype(float) - Xrecovery[:, :, 0], 'fro') ** 2 / (N[0] * N[1])
        mseC2 = np.linalg.norm(I[:, :, 1].astype(float) - Xrecovery[:, :, 1], 'fro') ** 2 / (N[0] * N[1])
        mseC3 = np.linalg.norm(I[:, :, 2].astype(float) - Xrecovery[:, :, 2], 'fro') ** 2 / (N[0] * N[1])
        iter += 1
        print(f"Epoch = {iter}")
        tol = np.linalg.norm((X - last_ten)) / train_norm
        last_ten = X.copy()
        if (tol < epsilon) or (iter >= maxiter):
            break

    return Xori, rtensor + np.mean(train_matrix), M + np.mean(train_matrix)



        





