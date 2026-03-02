import cupy as np
import numpy
from cupyx.scipy.sparse import linalg, LinearOperator
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


def local_admm(lambda, Kd, Kt, Ks, pos_obs, YR_tilde, priorvalue, max_iter):
    d1, d2, d3 = YR_tilde.shape
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs]
    x = priorvalue.copy()

    #---------- admm ----------
    for j in range(max_iter):
        z_prev = z.copy()
        #---------- r-update ----------
        ar = local_operator(x, pos_obs, Kd, Kt, Ks, lambda, d1, d2, d3)

    



def qktf(X, mask_data, R, psi, tau, sigma, K0):
    N = X.shape # gets the shape of the data
    N = np.array(N) # sets the shape of the data to an array
    D = X.ndim # gets the dimensions of the data

    #---------- Binary indicator matrix ----------
    mask_data = mask_data.astype(bool) # sets data to True if observed, False if not observed
    non_obs = np.where(mask_data == 0) # variable for unobserved data
    mask_matrix = [unfold(X, d) for d in range(D)] # unfolds binary matrix
    col = np.sum(mask_matrix[2], axis = 0) > 0 # returns boolean mask where columns of mask_matrix contain at least one zero
    train_mat = X * mask_data # sets only observed data in actual dataset
    mask = train_mat > 0 # creates boolean mask of same shape as train_mat
    train_mat = train_mat[mask] # whenever mask is True keep data point, if False discard - creates 1-D array of positive entries
    X_centred = X - np.mean(train_mat) # removes global mean before modelling - ADMM converges faster
    obs_centred = X_centred * mask_data # keeps mean-centred values for observed data
    mask_matrix_c = [mask_matrix[d].T for d in range(D)] # creates O_(d)^T
    vec_mask = [mask_matrix[d].ravel(order = 'F') for d in range(D)] # creates vec(O_(d)^T)
    obs = [np.where(vec_mask[d] == 1) for d in range(D)] # where data is observed in vec(O_(d)^T)
    sum_obs = np.sum(mask_data == 1) # sum of observed values

    #---------- Initialisations ----------
    theta = np.zeros(N) # Initialise theta as 0
    z = np.zeros(N) # Initialise z as 0
    U = np.zeros(N) # Initialise latent matrices to 0
    rtensor = np.zeros(N) # Initialises r to 0
    y = np.zeros(N) # Initialises y to 0
    x = np.zeros(N) # Initialises x to 0
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rvector = rtensor.ravel(order='F')
    m_unfold = U[0] @ khatri_rao(U[2], U[1]).T
    obs_centred[non_obs] = m[non_obs] + rtensor[non_obs]

    d_all = np.arange(D) # array of all dimensions
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

        





