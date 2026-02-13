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

def prox_map(xi, alpha, tau):
        """Proximal mapping = xi - max((tau - 1) / alpha, min(xi, tau/alpha))
        """
        low = (tau - 1) / alpha
        high = tau / alpha
        return xi - np.maximum((tau - 1) / alpha, np.minimum(xi, tau / alpha))    

def make_op(Hd, mask_matrix_c, psi, sigma, kdu, R, M):
    Hd_T = Hd.T
    n = R*M

    def matvec(v):
        V = v.reshape(R, M, order='F')
        HV = Hd @ V # N * M matrix (M = I_d)
        HV *= mask_matrix_c # Projector for mask matrix
        HtHV = Hd_T @ HV # R * M
        cov = psi * (V @ kdu) # R * M
        op = (sigma * HtHV) + cov # Operator for CG method
        return op.ravel(order='F') 
    
    return LinearOperator((n, n), matvec=matvec, dtype=float)

def global_admm(kdu, psi, sigma, g_mat, Hd, mask_matrix_c, x, z, sum_obs, priorvalue, max_iter, tau):
    """
    Function that calculate the CG for U_d update in ADMM function
    
    :param kdu: covariance norm (kernel)
    :param psi: regularisation parameter
    :param R: pre-specified rank
    :param sigma: regularisation parameter
    :param G: tensor Y - R
    """
    R, M = g_mat.shape
    n = R * M
    Hd_T = Hd.T
    rhs_mat = mask_matrix_c * (z - g_mat - x)
    b_mat = sigma * (Hd_T @ rhs_mat)
    b = b_mat.ravel(order='F')

    #---------- u-update ----------
    au = make_op(Hd, mask_matrix_c, psi, sigma, kdu, R, M)
    x0 = priorvalue.copy()
    u_vec, info = linalg.cg(au, b, x0 = x0, atol = 1e-4, maxiter = max_iter)
    
    #---------- z-update (proximal mapping) ----------
    alpha = sum_obs * sigma
    xi = g_mat - x - (au @ u_vec) 

    z = prox_map(xi, alpha)
    
    #---------- x-update (lagrangian multiplier) ----------
    x = x + (au @ u_vec) + z - g_mat

    return u_vec, z, x, info

def qktf(X, mask_data, R, psi, tau, sigma, K0):
    N = X.shape # gets the shape of the data
    N = np.array(N) # sets the shape of the data to an array
    D = X.ndim # gets the dimensions of the data
    
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
    x = np.zeros((N)) # Initialise theta as 0
    z = np.zeros((N)) # Initialise z as 0
    U = np.zeros((N)) # Initialise latent matrices to 0
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]
    rtensor = np.zeros(N)
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

        





