import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.linalg import khatri_rao
from cupyx.scipy.sparse import linalg, eye, csr_matrix
import numpy
from tqdm import tqdm

def cov_matern(d, loghyper, x):
    ell = cp.exp(loghyper[0])
    sf2 = cp.exp(2 * loghyper[1])
    def f(t):
        if d == 1: return 1
        if d == 3: return 1 + t
        if d == 5: return 1 + t * (1 + t / 3)
        if d == 7: return 1 + t * (1 + t * (6 + t) / 15)
    def m(t):
        return f(t) * cp.exp(-t)
    dist_sq = ((x[:, None] - x[None, :]) / ell) ** 2
    return sf2 * m(cp.sqrt(d * dist_sq))

def bohman(loghyper, x):
    range_ = cp.exp(loghyper[0])
    dis = cp.abs(x[:, None] - x[None, :])
    r = cp.minimum(dis / range_, 1)
    k = (1 - r) * cp.cos(cp.pi * r) + cp.sin(cp.pi * r) / cp.pi
    k[k < 1e-16] = 0
    k[cp.isnan(k)] = 0
    return k

def unfold(tensor, mode):
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
    eigvals_clipped = cp.maximum(eigvals, floor)
    K_psd = eigvecs @ cp.diag(eigvals_clipped) @ eigvecs.T # reconstructs from the clipped spectrum.

    return K_psd

def kronecker_mvm(Kr, vec, shape):
    D = len(shape) 
    x = vec.reshape(shape, order = 'F') 

    for d in range(D): 
        x_unfold = unfold(x, d) 
        x_unfold = Kr[d] @ x_unfold 
        x = fold(x_unfold, shape, d) 
    
    return x.ravel(order = 'F')

def build_khatri_rao(U, dims):
    dims = [int(d) for d in dims] 
    if len(dims) == 1:
        return U[dims[0]]
    else:
        result = U[dims[-1]]
        for i in range(len(dims) - 2, -1, -1):
            result = khatri_rao(result, U[dims[i]])
        return result

def reconstruct_tensor(U, shape):
    D = len(shape)
    dims_except_0 = list(range(1, D)) 
    if len(dims_except_0) > 0: 
        KrU = build_khatri_rao(U, dims_except_0) 
        M_unfold = U[0] @ KrU.T
    else:
        M_unfold = U[0] 

    M = M_unfold.reshape(shape, order = 'F') 
    return M

def Ap_operatorT(vec, maskT, KrU, KrU_T, Qu, rho, R, M):
    X = vec.reshape(R, M, order = 'F')
    temp = KrU @ X
    temp *= maskT
    Ap1 = KrU_T @ temp
    Ap2 = rho * (X @ Qu)
    return (Ap1 + Ap2).ravel(order = 'F')

def cg_factorT(Qu, rho, KrU, mask_matrixT, YR_tilde, priorvalue, max_iter):
    R, M = YR_tilde.shape
    n = R * M
    b = YR_tilde.ravel(order = 'F')
    KrU_T = KrU.T
    def matvec(v):
        return Ap_operatorT(v, mask_matrixT, KrU, KrU_T, Qu, rho, R, M)
    A_op = LinearOperator((n, n), matvec=matvec, dtype=b.dtype)
    x0 = priorvalue.copy()
    x, info = linalg.cg(A_op, b, x0=x0, atol=1e-4, maxiter=max_iter)
    return x, info

def Ap_operatorL(vec, pos_obs, Kr, gamma, N):
    x = cp.zeros(int(numpy.prod(N)))
    x[pos_obs] = vec
    Ap1 = kronecker_mvm(Kr, x, N)
    return Ap1[pos_obs] + gamma * vec

def cg_local(gamma, Kr, pos_obs, YR_tilde, priorvalue, max_iter):
    N = numpy.array(YR_tilde.shape)
    n_obs = pos_obs[0].shape[0]
    b = (YR_tilde.ravel(order = 'F'))[pos_obs]
    def matvec(v):
        return Ap_operatorL(v, pos_obs, Kr, gamma, N)
    A_op = LinearOperator((n_obs, n_obs), matvec=matvec, dtype=b.dtype)
    x0 = priorvalue.copy()
    x_gpu, info = linalg.cg(A_op, b, x0=x0, atol=1e-4, maxiter=max_iter)
    return x_gpu, info

def GLSKF(I, Omega, lengthscaleU: list, lengthscaleR: list, varianceU: list, varianceR: list, tapering_range, d_MaternU, d_MaternR, R, rho, gamma, maxiter, K0, epsilon, distance_matrix=None, seed=None):

    N = I.shape
    N = numpy.array(N)

    D = I.ndim
    maxP = float(cp.max(I))

    Omega = Omega.astype(bool)
    pos_miss = cp.where(Omega == 0)
    num_obser = cp.sum(Omega)
    mask_matrix = [unfold(Omega, d) for d in range(D)]
    idx = cp.sum(mask_matrix[D-1], axis = 0) > 0
    train_matrix = I * Omega
    train_matrix = I[Omega]

    centre = cp.mean(train_matrix)
    Isubmean = I - centre
    T = Isubmean * Omega
    mask_matrixT = [mask_matrix[d].T for d in range(D)]
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)]
    pos_obs = [cp.where(mask_flat[d] == 1) for d in range(D)]

    hyper_Ku, hyper_Kr = [None] * D, [None] * D
    Ku, Kr = [None] * D, [None] * D
    invKu = [None] * D

    for d in range(D-1):
        if distance_matrix is not None and distance_matrix[d] is not None:
            x = distance_matrix[d]
        else:
            x = cp.arange(N[d]) # creates a vector of integers from 1 to the size of the current dimension - used as input for the covariance function.
        hyper_Ku[d] = [cp.log(lengthscaleU[d]), cp.log(varianceU[d])]
        Ku[d] = cov_matern(d_MaternU, hyper_Ku[d], x)
        invKu[d] = cp.linalg.inv(Ku[d])

        hyper_Kr[d] = [cp.log(lengthscaleR[d]), cp.log(varianceR[d]), cp.log(tapering_range)]
        TaperM = bohman([hyper_Kr[d][2]], x)
        Kr[d] = csr_matrix(cov_matern(d_MaternR, hyper_Kr[d][:2], x) * TaperM)

    invKu[D-1] = cp.eye(N[D-1])
    Kr[D-1] = csr_matrix(cp.eye(N[D-1]))

    rng = numpy.random.default_rng(seed)

    X = T.copy()
    X[pos_miss] = T.sum() / num_obser
    U = [cp.asarray(0.1 * rng.standard_normal((N[d], R))) for d in range(D)]
    M = reconstruct_tensor(U, N)
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    Rtensor = cp.zeros(N)
    Rvector = Rtensor.ravel(order = 'F')
    Rvector_temp = Rtensor.ravel(order = 'F')
    X[pos_miss] = M[pos_miss] + Rtensor[pos_miss]

    d_all = cp.arange(0, D)
    train_norm = cp.linalg.norm(T)
    last_ten = T.copy()
    approxU = [None] * D
    pbar = tqdm(total=maxiter, desc="GLSKF Iterations") # creates a progress bar for the ADMM iterations.
    iter = 0
    while True:
        Gtensor = X - Rtensor
        Gtensor_mask = Gtensor * Omega
        for d in range(D):
            dsub = cp.delete(d_all, d)
            dsub = numpy.array(dsub.get())
            KrU = build_khatri_rao(U, dsub)
            HG = KrU.T @ unfold(Gtensor_mask, d).T
            UTvector[d], approxU[d] = cg_factorT(invKu[d], rho, KrU, mask_matrixT[d], HG, UTvector[d], 500)
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T
        M = reconstruct_tensor(U, N)
        X[pos_miss] = M[pos_miss] + Rtensor[pos_miss]
        if iter >= K0:
            Ltensor = X - M
            Ltensor_mask = Ltensor * Omega
            Rvector_temp[pos_obs[0]], approxE = cg_local(gamma, Kr, pos_obs[0], Ltensor_mask, Rvector_temp[pos_obs[0]], 500)
            Rvector = kronecker_mvm(Kr, Rvector_temp, N)
            Rtensor = Rvector.reshape(N, order = 'F')
            Rtensor_unfold = unfold(Rtensor, D-1)
            Rtensor_unfold_obs = Rtensor_unfold[:, idx]
            Kr_raw = cp.cov(Rtensor_unfold_obs) # calculates the new covariance in the last dimension.
            Kr_psd = enforce_psd(Kr_raw)
            x_last = cp.arange(1, N[D-1] + 1)
            TaperM_last = bohman([cp.log(tapering_range)], x_last)
            Kr_tapered = Kr_psd * TaperM_last
            Kr[D-1] = csr_matrix(Kr_tapered)
        else:
            Rtensor[:] = 0.0

        X[pos_miss] = M[pos_miss] + Rtensor[pos_miss]
        Xori = X + centre
        
        iter += 1
        tol = cp.linalg.norm((X - last_ten)) / train_norm
        last_ten = X.copy()

        pbar.update(1)

        if (tol < epsilon and iter > K0) or (iter >= maxiter):
            pbar.close()
            break
        
    return Xori, Rtensor, M + centre