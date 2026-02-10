import cupy as np
import numpy

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

def global_admm(u, lambda, rho, kdu, mask, tau, max_iter):
    """
    ADMM algorithm for estimating the global structure of data tensor
    
    :param u: latent matrices
    :param lamda: regulariser
    :param rho: regulariser
    :param kdu: covariance matrix
    :param mask: mask matrix
    :param tau: quantile level
    :param max_iter: maximum number of iterations
    """
    for k in range(max_iter):
        # latent matrix U_d^(k+1) update
        def shrink(x, alpha):
            """Shrink[x, alpha] = sgn(x) * max(abs(x) - alpha, 0)
            """
            return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        
    x, s, vh = np.linalg.svd(u) # svd to compute singular values for computation of eta (largest eigenvalues of u^T*u)
    eta = s[0]**2 # s[0] is the largest singular value

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






