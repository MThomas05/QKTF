import cupy as np
import numpy
from cupyx.scipy.sparse import linalg

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

def lin_operator(u, mask, Hd, Hd_T, psi, sigma, kdu, R, M):
    """Function that creates the linear operator used in conjugate gradient method
    """
    X = u.reshape(R, M, order = 'F') 
    A = Hd @ X
    A *= mask_T
    cov_A = psi * (Hd @ kdu)
    mask_a = sigma * (Hd_T @ A)
    A_op = (cov_A + mask_a).ravel(order = 'F')
    def matvec(v):
        return A_op
    AU = LinearOperator((n, n), matvec=matvec)
    return AU 


def global_admm(kdu, psi, R, sigma, G, Hd, Hd_T, mask, mask_T, theta, z, priorvalue, max_iter, tau):
    """
    Function that calculate the CG for U_d update in ADMM function
    
    :param kdu: covariance norm (kernel)
    :param psi: regularisation parameter
    :param R: pre-specified rank
    :param sigma: regularisation parameter
    :param G: tensor Y - R
    """
    # latent matrix u-update
    oh = mask @ Hd
    oh_T = mask_T @ Hd_T
    G_T = G.T
    a = psi * (kdu @ numpy.identity(R)) + (Hd @ mask)
    b1 = numpy.dot(theta, mask_T @ Hd)
    b2 = z * (oh_T + oh) - (Hd_T @ G.ravel(order='F') - Hd @ G_T.ravel(order='F'))
    b = b1 + 0.5*sigma*b2
    x0 = priorvalue.copy()
    def matvec(x):
        return 
    u, info = linalg.cg(a, b, x0=x0, atol = 1e-4, max_iter=max_iter)

    # auxiliary variable z-update
    def prox_map(xi, alpha):
        """Proximal mapping = xi - max((tau - 1) / alpha, min(xi, tau/alpha))
        """
        low = (tau - 1) / alpha
        high = tau / alpha
        return xi - numpy.maximum((tau - 1) / alpha, numpy.minimum(xi, tau / alpha))
    
    alpha = N * sigma
    xi = (mask @ G) - (oh @ u) + ((1 / sigma) * theta)

    z = prox_map(xi, alpha)
    
    # lagrangian multiplier theta-update

    theta = theta - sigma * (oh @ u.T) + z - (mask @ G)

    return u, info

def qktf(X, mask_data):
    N = X.shape # gets the shape of the data
    N = numpy.array(N) # sets the shape of the data to an array
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
    obs = [np.where(mask_flat[d] == 1) for d in range(D)] # where data is observed in vec(O_(d)^T)


    theta = 0 # Initialise theta as 0
    z = 0 # Initialise z as 0
    U = 0 # Initialise latent matrices to 0
    Uvector = [U[d].ravel(order = 'F') for d in range(D)]
    UTvector = [U[d].T.ravel(order = 'F') for d in range(D)]

    d_all = np.arange(D) # array of all dimensions
    Gtensor = X - Rtensor # G_Omega in latent matrix optimisation
    approxU = [None] * D
    iter = 0

    while True:
        Gtensor = X - Rtensor # G_Omega in latent matrix optimisation
        Gtensor_mask = Gtensor * mask_data

        for d in range(D):
            dsub = numpy.delete(d_all, d) # deletes d dimension
            dsub = numpy.array(dsub.get()) # creates an array and brings to CPU
            Hdu = khatri_rao(U[dsub[1]], U[dsub[0]]) # creates H_d^u with k and k+1 estimates of U_d
            Hdu_T = Hdu.T
            G = Hdu_T @ unfold(Gtensor_mask, d).T
            UTvector[d], approxU[d] = global_admm(kdu, psi, R, G, Hdu, Hdu_T, mask_matrix_c, mask_matrix_cT, theta, z, UTvector[d], 1000, tau)

        





