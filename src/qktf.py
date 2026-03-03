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

def global_admm(Qu, psi, sigma, KrU, mask_matrixT, YR_tilde, priorvalue, max_iter, tau, z, theta, sum_obs, total_data):
    R, M = YR_tilde.shape
    YR_flat = YR_tilde.ravel(order = 'F')
    x0 = priorvalue.copy()
    x0_flat = x0.ravel(order = 'F')
    z_flat = z.ravel(order = 'F')
    theta_flat = theta.ravel(order = 'F')
    KrU_T = KrU.T

    for j in range(max_iter):
        z_prev = z_flat.copy()
        temp_b = KrU @ x0
        temp_b *= mask_matrixT
        rhs_mat = mask_matrixT * (YR_tilde - z - theta)
        b_mat = sigma * (KrU_T @ rhs_mat)
        b = b_mat.ravel(order='F')
        #---------- u-update ----------
        A_op = LinearOperator((R*M), (R*M), matvec=lambda v: global_operator(v, mask_matrixT, KrU, KrU_T, Qu, psi, sigma, R, M))
        u_vec, info = linalg.cg(A_op, b, x0 = x0_flat, atol = 1e-4, maxiter = max_iter)
    
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
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm((mask_matrixT @ KrU_T) * theta)
        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break    

    return u_vec, z_vec, theta, info
    
def local_operator(vec, pos_obs, Kd, Kt, Ks, lambda, d1, d2, d3):
    x = np.zeros(d1, d2, d3)
    x[pos_obs] = vec
    Ap = kronecker_mvm(Kd, Kt, Ks, x, d1, d2, d3)
    return Ap[pos_obs] + lambda * vec

def local_admm(lambda, r, a, v, Kd, Kt, Ks, pos_obs, sum_obs, YR_tilde, priorvalue, mask_matrixT, max_iter, tau, total_data):
    d1, d2, d3 = YR_tilde.shape
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs]
    r = priorvalue.copy()
    r_flat = r.ravel(order = 'F')
    a_vec = a.ravel(order = 'F')
    v_vec = v.ravel(order = 'F')
    N = d1 * d2 * d3

    #---------- admm ----------
    for j in range(max_iter):
        a_prev = a_vec.copy()
        rhs_mat = Y_obs - a_vec - v_vec
        b = np.zeros(N)
        b[pos_obs] = lambda * rhs_mat
        #---------- r-update ----------
        ar = LinearOperator((N, N), matvec=lambda v: local_operator(v, pos_obs, Kd, Kt, Ks, lambda, d1, d2, d3))
        r_vec, info = linalg.cg(ar, b, x0 = r_flat, atol = 1e-4, maxiter = max_iter)

        #---------- z-update ---------
        r_obs = r_vec[pos_obs]
        zeta = Y_obs - v_vec - r_obs
        alpha = sum_obs * lambda

        a_vec = prox_map(zeta, alpha, tau)

        #---------- x-update ----------
        v_vec = v_vec + r_obs + a_vec - Y_obs

        #---------- convergence criterion ----------
        res_pri = r_obs + a_vec - Y_obs
        res_dual = lambda * (a_vec - z_prev)
        eps_pri = np.sqrt(sum_obs) * 1e-4 + 1e-4 * np.maximum(np.maximum(np.linalg.norm(r_obs), np.linalg.norm(a_vec)), np.linalg.norm(Y_obs))
        eps_dual = np.sqrt(total_data) * 1e-4 + 1e-4 * np.linalg.norm(v_vec)

        if np.linalg.norm(res_pri) <= eps_pri and np.linalg.norm(res_dual) <= eps_dual:
            break

    return r_vec, a_vec, v_vec, info

def qktf(I, Omega, lengthscaleU: list, lengthscaleR: list, varianceU: list, varianceR: list, tapering_range, d_maternU, d_maternR, R, psi, sigma, lambda, tau, maxiter, K0, epsilon):
    tensor_shape = I.shape
    N = tensor_shape
    D = len(N)
    total_data = np.prod(N)
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
    hyper_Kr[0] = [np.log(lengthscaleR[0]), np.log(varianceR[0]), np.log(tapering_range)]
    hyper_Kr[1] = [np.log(lengthscaleR[1]), np.log(varianceR[1]), np.log(tapering_range)]

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
    theta = [np.zeros(np.sum(mask_matrix[d])) for d in range(D)] # Initialise theta as 0
    z = [np.zeros(np.sum(mask_matrix[d])) for d in range(D)] # Initialise z as 0
    U = [[np.zeros((R, N[d])) for d in range(D)]] # Initialise latent matrices to 0
    rtensor = np.zeros(N) # Initialises r to 0
    y = np.zeros(N) # Initialises y to 0
    v = np.zeros(N) # Initialises v to 0
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rvector = rtensor.ravel(order='F')
    rvector_temp = rtensor.ravel(order = 'F')
    M_unfold1 = U[0] @ khatri_rao(U[2], U[1]).T
    M = fold(M_unfold1, N, 0)
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
            dsub = numpy.array(dsub.get())
            KrU = khatri_rao(U[dsub[1]], U[dsub[0]])
            HG = KrU.T @ unfold(Gtensor_mask, d).T
            UTvector[d], z[d], theta[d], info = global_admm(invKu[d], psi, sigma, KrU, mask_matrixT[d], HG, UTvector[d], 100, tau, z[d], theta[d], num_obs)
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T
        M_unfold1 = U[0] @ (khatri_rao(U[2], U[1]).T)
        M = fold(M_unfold1, N, 0)
        X[pos_miss] = M[pos_miss] + rtensor[pos_miss]
        if iter >= K0:
            Ltensor = X - M
            Ltensor_mask = Ltensor * Omega
            rvector_temp[pos_obs[0]], approxE = local_admm(lambda, x, a, Kr[2], Kr[1], Kr[0], pos_obs[0], num_obs, Ltensor_mask, mask_matrixT, rvector_temp[pos_obs[0]], 100, tau)
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
        psnrC1 = 10 * np.log10(maxP ** 2 / mseC1)
        mseC2 = np.linalg.norm(I[:, :, 1].astype(float) - Xrecovery[:, :, 1], 'fro') ** 2 / (N[0] * N[1])
        psnrC2 = 10 * np.log(maxP ** 2 / mseC2)
        mseC3 = np.linalg.norm(I[:, :, 2].astype(float) - Xrecovery[:, :, 2], 'fro') ** 2 / (N[0] * N[1])
        psnrC3 = 10 * np.log(maxP ** 2 / mseC3)
        psnrf[iter] = (psnrC1 + psnrC2 + psnrC3) / 3
        iter += 1
        print(f"Epoch = {iter}, PSNR = {psnrf[iter-1]}")
        tol = np.linalg.norm((X - last_ten)) / train_norm
        last_ten = X.copy()
        if (tol < epsilon) or (iter >= maxiter)
            break

    return Xori, rtensor + np.mean(train_matrix), M + np.mean(train_matrix)



        





