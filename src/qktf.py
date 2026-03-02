import cupy as np
import numpy
from cupyx.scipy.sparse import linalg, LinearOperator, eye, csr_matrix
from cupyx.scipy.linalg import khatri_rao

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
    Function that performs a mode-d unfolding of a tensor.
    
    :param tensor: input tensor
    :param mode: axis to do unfolding
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def fold(mat, dim, mode):
    """
    Function that performs folding of an unfolded tensor.
    
    :param mat: unfolded tensor
    :param dim: tuple/1d array containing dimensions of original tensor
    :param mode: axis where tensor was unfolded
    """
    index = list() # defines new axis order - encodes which axes became rows
    index.append(mode) # ensures 'mode' axis is first
    for i in range(dim.shape[0]): # appends all other axes in order
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

def kronecker_mvm(K3, K2, K1, vec, d1, d2, d3):
    temp1 = (K1 @ vec.reshape(d1, d2, d3, order = 'F').reshape(d1, -1)).reshape(d1, d2, d3)
    temp2 = (K2 @ temp1.transpose(1, 0, 2).reshape(d2, -1)).reshape(d2, d1, d3).transpose(1, 0, 2)
    temp3 = (K3 @ temp2.transpose(2, 0, 1).reshape(d3, -1)).reshape(d3, d1, d2).transpose(1, 2, 0)
    return temp3.ravel(order = 'F')

def prox_map(xi, alpha, tau):
        """Proximal mapping = xi - max((tau - 1) / alpha, min(xi, tau/alpha))
        """
        low = (tau - 1) / alpha
        high = tau / alpha
        return xi - np.maximum((tau - 1) / alpha, np.minimum(xi, tau / alpha))    

def global_operator(vec, maskT, KrU, KrU_T, Qu, psi, sigma, R, M):
    X = vec.reshape(R, M, order = 'F')
    temp = KrU @ X
    temp *= maskT
    Ap1 = sigma * (KrU_T @ temp)
    Ap2 = psi * (X @ Qu)
    return (Ap1 + Ap2).ravel(order = 'F')

def global_admm(Qu, psi, sigma, KrU, mask_matrixT, vec_mask, YR_tilde, priorvalue, max_iter, tau, z, theta, sum_obs):
    R, M = YR_tilde.shape
    YR_flat = YR_tilde.ravel(order = 'F')
    x0 = priorvalue.copy()
    x0_flat = x0.ravel(order = 'F')
    z_flat = z.ravel(order = 'F')
    theta_flat = theta.ravel(order = 'F')
    KrU_T = KrU.T

    for j in range(max_iter):
        z_prev = z_vec.copy()
        temp_b = KrU @ x0
        temp_b *= mask_matrixT
        rhs_mat = mask_matrixT * (YR_tilde - z - theta)
        b_mat = sigma * (KrU_T @ rhs_mat)
        b = b_mat.ravel(order='F')
        #---------- u-update ----------
        A_op = LinearOperator((R*M), (R*M), matvec=lambda v: global_operator(v, mask_matrixT, KrU, KrU_T, Qu, psi, sigma, R, M))
        u_vec, info = linalg.cg(A_op, b, x0 = x0_flat, atol = 1e-4, maxiter = max_iter)
        u_mat = u_vec.reshape(R, M, order = 'F')
    
        #---------- z-update (proximal mapping) ----------
        alpha = sum_obs * sigma
        eta = YR_flat - theta_flat - u_vec

        z_vec = prox_map(eta, alpha, tau)
    
        #---------- x-update (lagrangian multiplier) ----------
        theta = theta_flat + u_vec + z_vec - YR_flat

        #---------- residuals ----------
        res_pri = u_vec + z_vec - (mask_matrixT * YR_flat)
        res_dual = (sigma * temp_b) * (z_vec - z_prev)
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(u_vec), np.linalg.norm(z)), np.linalg.norm(YR_tilde))
        eps_dual = np.sqrt(R) * 1e-4 + 1e-4 * np.linalg.norm((mask_matrixT @ KrU_T) * theta)
        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break    

    return u_vec, z_vec, theta, info
    
def local_operator(vec, pos_obs, Kd, Kt, Ks, lambda, d1, d2, d3):
    x = np.zeros(d1, d2, d3)
    x[pos_obs] = vec
    Ap = kronecker_mvm(Kd, Kt, Ks, x, d1, d2, d3)
    return Ap[pos_obs] + lambda * vec

def local_admm(lambda, x, theta, Kd, Kt, Ks, pos_obs, sum_obs, YR_tilde, priorvalue, mask_matrixT, max_iter, tau):
    d1, d2, d3 = YR_tilde.shape
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs]
    x = priorvalue.copy()
    x_flat = x.ravel(order = 'F')
    z_flat = z.ravel(order = 'F')

    #---------- admm ----------
    for j in range(max_iter):
        z_prev = z.copy()
        rhs_mat = mask_matrixT * (y - z - x)
        b_mat = lambda * rhs_mat
        b = b_mat.ravel(order='F')
        #---------- r-update ----------
        ar = LinearOperator((R*M), (R*M), matvec=lambda v: local_operator(v, pos_obs, Kd, Kt, Ks, lambda, d1, d2, d3))
        r_vec, info = linalg.cg(ar, b, x0 = x_flat,atol = 1e-4, maxiter = max_iter)

        #---------- z-update ---------
        zeta = Y_obs - x_flat - (mask_matrixT * r_vec)
        alpha = sum_obs * lambda

        z_vec = prox_map(zeta, alpha, tau)

        #---------- x-update ----------
        x = x_flat + (mask_matrixT * r_vec) + z_vec - Y_obs

        #---------- convergence criterion ----------
    return r_vec, z_vec, x, info

def qktf(I, Omega, lengthscaleU: list, lengthscaleR: list, varianceU: list, varianceR: list, tapering_range, d_maternU, d_maternR, psi, sigma, lambda, tau, maxiter, K0, epsilon):
    N = I.shape # gets the shape of the data
    N = numpy.array(N) # sets the shape of the data to an array
    D = I.ndim # gets the dimensions of the data
    maxP = float(np.max(I))

    #---------- Binary indicator matrix ----------
    Omega = Omega.astype(bool)
    pos_miss = np.where(Omega == 0)
    num_obs = np.sum(Omega)
    mask_matrix = [unfold(Omega, d) for d in range(D)]
    idx = np.sum(mask_matrix[2], axis = 0) > 0
    train_matrix = I * Omega
    train_matrix = train_matrix[train_matrix > 0]
    Isubmean = I - np.mean(train_matrix)
    T = Isubmean * Omega
    mask_matrixT = [mask_matrix[d].T for d in range(D)]
    mask_flat = [mask_matrix[d].ravel(order = 'F') for d in range(D)]
    pos_obs = [np.where(mask_flat[d] == 1) for d in range(D)]

    #---------- Covariance constraints ----------
    hyper_Ku = [None] * D
    hyper_Ku[0] = [np.log(lengthscaleU[0]), np.log(varianceU[0])]
    hyper_Ku[1] = [np.log(lengthscaleU[1]), np.log(varianceU[1])]
    
    hyper_Kr = [None] * D
    hyper_Kr = [np.log(lengthscaleR[0]), np.log(varianceR[0]), np.log(tapering_range)]
    hyper_Kr = [np.log(lengthscaleR[1]), np.log(varianceR[1]), np.log(tapering_range)]

    Ku, Kr = [None] * D, [None] * D
    invKu = [None] * D

    x = np.arange(1, N[0] + 1)
    Ku[0] = cov_matern(d_maternU, hyper_Ku[0], x)
    invKu[0] = np.linalg.inv(Ku[0])
    TaperM = bohman([hyper_Ku[1][2]], x)
    Kr[1] = csr_matrix(cov_matern(d_maternR, hyper_Ku[0][:2], x) * TaperM)

    x = np.arange(1, N[1] + 1)
    Ku[1] = cov_matern(d_maternU, hyper_Ku[0], x)
    invKu = np.linalg.inv(Ku[1])
    TaperM = bohman([hyper_Ku[1][2]], x)
    Kr[1] = csr_matrix(cov_matern(d_maternR, hyper_Kr[1][:2], x) * TaperM)

    invKu[2] = csr_matrix(eye(N[2]))
    Kr[2] = csr_matrix(eye(N[2]))

    #---------- Initialisations ----------
    X = T
    X[pos_miss] = T.sum() / num_obs
    theta = np.zeros(N) # Initialise theta as 0
    z = np.zeros(N) # Initialise z as 0
    U = np.zeros(N) # Initialise latent matrices to 0
    rtensor = np.zeros(N) # Initialises r to 0
    y = np.zeros(N) # Initialises y to 0
    x = np.zeros(N) # Initialises x to 0
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rvector = rtensor.ravel(order='F')
    M_unfold1 = U[0] @ khatri_rao(U[2], U[1]).T
    M = fold(M_unfold1, N, 0)
    X[pos_miss] = M[pos_miss] + rtensor[pos_miss]
    d_all = np.arange(D) # array of all dimensions
    train_norm = np.linalg.norm(T)
    approxU = [None] * D
    iter = 0

    while True:
        Gtensor = X - rtensor # G_Omega in latent matrix optimisation
        Gtensor_mask = Gtensor * mask_data

        for d in range(D):
            dsub = numpy.delete(d_all, d) # deletes d dimension
            dsub = numpy.array(dsub.get()) # creates an array and brings to CPU
            Hdu = khatri_rao(U[dsub[1]], U[dsub[0]]) # creates H_d^u with k and k+1 estimates of U_d
            g_mat = Hdu.T @ unfold(Gtensor_mask, d).T
            UTvector[d], approxU[d] = global_admm(kdu, psi, g_mat, Hdu, mask_matrix_c[d], x, z, UTvector[d], 1000, tau)
            U[d] = (UTvector[d].reshape(R, N[d], order='F')).T

        m_unfold1 = U[0] @ (khatri_rao(U[2], U[1]).T)
        m = fold(m_unfold1, N, 0)
        obs_centred[non_obs] = m[non_obs] + rtensor[non_obs]

        





