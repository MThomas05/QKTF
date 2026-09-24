import cupy as cp
import numpy
from tqdm import tqdm
from cupyx.scipy.linalg import khatri_rao
from cupyx.scipy.sparse import linalg, eye, csr_matrix

def cov_matern(d, loghyper, x):
    """Compute a Matern covariance matrix.

    Args:
        d (int): Matern smoothness parameter.
        loghyper (cupy.ndarray): Log length-scales and log variances
        x (cupy.ndarray): Co-ordinates or precomputed pairwise distance matrix.

    Returns:
        cupy.ndarray: Matern covariance matrix.
    """
    ell = cp.exp(loghyper[0])
    sf2 = cp.exp(2*loghyper[1])

    def f(t):
        if d == 1:
            return 1
        if d == 3:
            return 1 + t
        if d == 5:
            return 1 + t*(1 + t/3)
        if d == 7:
            return 1 + t*(1 + t*(6 + t)/15)
        
    def m(t):
        return f(t)*cp.exp(-t)

    if x.ndim == 2:
        dis = x
    else:
        dis = cp.abs(x[:, None] - x[None, :])

    dist_sq = (dis / ell)**2

    return sf2 * m(cp.sqrt(d*dist_sq))
                                                                               
def bohman(loghyper, x):
    """Compute a Bohman covariance taper matrix.

    Args:
        loghyper (cupy.ndarray): Log tapering range.
        x (cupy.ndarray): Co-ordinates or precomputed pairwise distance matrix.

    Returns:
        cupy.ndarray: Bohman taper matrix.
    """
    range_ = cp.exp(loghyper[0])

    if x.ndim == 2:
        dis = x
    else:
        dis = cp.abs(x[:, None] - x[None, :])

    r = cp.minimum(dis/range_, 1)
    k = (1 - r)*cp.cos(cp.pi*r) + cp.sin(cp.pi*r)/cp.pi
    k[k < 1e-16] = 0
    k[cp.isnan(k)] = 0

    return k

def unfold(tensor, mode):
    """Perform mode-d unfolding of a tensor.

    Inputs:
        tensor (cupy.ndarray): Tensor to be unfolded.
        mode (int): Mode along which to unfold the tensor.

    Returns:
        cupy.ndarray: Unfolded tensor.
    """
    return cp.reshape(cp.moveaxis(tensor, mode, 0),
                      (tensor.shape[mode], -1), 
                      order='F')

def fold(mat, dims, mode):
    """Perform mode-d folding of a matrix.

    Args:
        mat (cupy.ndarray): Matrix to be folded.
        dims (ndarray): Dimensions of the original tensor.
        mode (int): Mode along which to fold the matrix.

    Returns:
        cupy.ndarray: Folded tensor.
    """
    index = [mode] + [i for i in range(dims.shape[0]) if i != mode]
    return cp.moveaxis(cp.reshape(mat, list(dims[index]), order='F'),
                       0, 
                       mode)

def build_khatri_rao(U, modes):
    """Build the Khatri-Rao product of selected latent matrices.

    Args:
        U (list): List of latent factor matrices.
        modes (cupy.ndarray): Indices of the modes to include in the product.

    Returns:
        cupy.ndarray: Khatri-Rao product of the selected latent matrices.
    """
    modes = [int(mode) for mode in modes]

    if len(modes) == 1:
        return U[modes[0]]
    else:
        result = U[modes[-1]]

        for i in range(len(modes) - 2, -1, -1):
            result = khatri_rao(result, U[modes[i]])

        return result

def reconstruct_tensor(U, shape):
    """Reconstruct the global tensor component from its CP factor matrices.

    Args:
        U (list): List of latent factor matrices.
        shape (tuple): Shape of the tensor.

    Returns:
        cupy.ndarray: Reconstruct global component.
    """
    n_modes = len(shape) 
    dims_except_0 = list(range(1, n_modes)) 

    if len(dims_except_0) > 0: 
        kr_u = build_khatri_rao(U, dims_except_0) 
        m_unfold = U[0] @ kr_u.T 
    else:
        m_unfold = U[0] 

    m = m_unfold.reshape(shape, order = 'F') 

    return m


def prox_map(xi, alpha, tau):
    """Apply the proximal operator for the ADMM z-update.

    Args:
        xi (cupy.ndarray): Input vector.
        alpha (float): Proximal parameter.
        tau (float): Quantile parameter.

    Returns:
        cupy.ndarray: Vector after applying the proximal operator.
    """
    low = (tau - 1)/alpha 
    high = tau/alpha 

    return xi - cp.maximum(low, cp.minimum(xi, high)) 

def global_operator(vec, mask_t, kr_u, kr_u_t, 
                    q_u, psi, sigma, rank, mode_size):
    """Apply the global linear operator used by CG solver.

    Args:
        vec (cupy.ndarray): Vector to be multiplied by the global operator.
        mask_T (cupy.ndarray): Boolean mask of observed tensor entries.
        kr_u (cupy.ndarray): Khatri-Rao product of the latent factor matrices.
        kr_u_t (cupy.ndarray): Transpose of the Khatri-Rao product.
        q_u (cupy.ndarray): Covariance regulated matrix for the current mode.
        psi (float):  Global regularisation parameter.
        sigma (float): ADMM penalty parameter.
        rank (int): CP rank. 
        mode_size (int): Size of the current tensor mode.
        verbose (bool): Whether to print operator diagnostics

    Returns:
        cupy.ndarray: Result of applying the global linear operator to 'vec'.
    """
    x = vec.reshape(rank, mode_size, order='F') 
    temp = kr_u @ x
    temp *= mask_t
    a_p1 = sigma*(kr_u_t @ temp) 
    a_p2 = (psi / (rank * mode_size))*(x @ q_u) 

    return (a_p1 + a_p2).ravel(order='F')

def global_admm(q_u, kr_u, mask_matrix_t, yr_tilde, priorvalue, z, theta,
                psi, sigma, inner_maxiter, tau, rank, sum_obs, cg_maxiter, 
                rel_tol, abs_tol, verbose=False):
    """Update a latent factor matrix using the ADMM algorithm.

    Args:
        q_u (cupy.ndarray): Covariance regulated matrix for the current mode.
        kr_u (cupy.ndarray): Khatri-Rao product of the latent factor matrices.
        mask_matrix_t (cupy.ndarray): Transposed boolean obervation matrix.
        yr_tilde (cupy.ndarray): Fixed tensor term for the current mode update.
        priorvalue (cupy.ndarray): Previous latent factor estimate used as the initial guess.
        z (cupy.ndarray): ADMM auxiliary variable.
        theta (cupy.ndarray): ADMM Lagrange multiplier variable.
        psi (float): Global regularisation parameter.
        sigma (float): ADMM penalty parameter.
        inner_maxiter (int): Maximum number of ADMM iterations.
        tau (float): Quantile parameter.
        rank (int): CP rank.
        sum_obs (int): Number of observed tensor entries.
        cg_maxiter (int): Maximum number of CG iterations.
        rel_tol (float): Relative convergence tolerance.
        abs_tol (float): Absolute convergence tolerance.
        verbose (bool): Whether to print optimisation diagnostics.

    Returns:
        u (cupy.ndarray): Updated latent factor vector.
        z (cupy.ndarray): Updated auxiliary variabel.
        theta (cupy.ndarray): Updated Lagrange multiplier variable.
        j + 1 (int): Number of ADMM iterations performed.
        converged (bool): Whether the ADMM algorithm converged.
        gcg_nonconverged (int): Number of non-converged CG solves.
    """
    mode_size = yr_tilde.shape[1] 
    x0 = priorvalue.copy() 
    kr_u_t = kr_u.T 
    dtype = yr_tilde.dtype

    def matvec(vec): 
        return global_operator(vec, mask_matrix_t, kr_u, kr_u_t, 
                               q_u, psi, sigma, rank, mode_size) 

    operator = linalg.LinearOperator((rank*mode_size, rank*mode_size),
                                     matvec=matvec, 
                                     dtype=dtype)
     
    assert inner_maxiter > 0, "global_admm requires at least one ADMM sweep."

    converged = False
    gcg_nonconverged = 0

    # ---- Diagnostics ----
    if verbose:
        print(f"z_norm={cp.linalg.norm(z)}",
              f"eta_norm={cp.linalg.norm(theta)}", 
              f"yr_tilde_norm={cp.linalg.norm(yr_tilde)}")

    # ----- ADMM -----
    for j in range(inner_maxiter):
        z_prev = z.copy()

        # ----- u-update -----
        bmat = sigma*(yr_tilde - z) + theta 
        bmat = kr_u_t @ (mask_matrix_t * bmat)
        b = bmat.ravel(order='F')
        u, info = linalg.cg(operator, b, x0=x0, atol=1e-4, maxiter=cg_maxiter) 

        if info != 0:
            gcg_nonconverged += 1

        if verbose:
            print(f"[global_admm] sweep {j}: CG info={info}"
                  f"u_norm={cp.linalg.norm(u)}")
            
        x0 = u 
        umat = u.reshape(rank, mode_size, order='F') 
        temp = kr_u @ umat 
        temp = mask_matrix_t*temp 

        # ----- z-update -----
        phi = yr_tilde - temp + (theta/sigma) 
        alpha = sum_obs * sigma 
        z = prox_map(phi, alpha, tau) 

        # ----- theta-update -----
        theta = theta - sigma*(temp + z - yr_tilde)

        # ----- Convergence checks -----
        res_pri = temp + z - yr_tilde 
        res_temp = kr_u_t @ (mask_matrix_t*(z - z_prev))
        res_dual = sigma*res_temp 

        eps_pri = (cp.sqrt(sum_obs)*abs_tol 
                   + rel_tol*max(cp.linalg.norm(temp), 
                                 cp.linalg.norm(z), 
                                 cp.linalg.norm(yr_tilde))) 
        eps_dual = (cp.sqrt(rank*mode_size)*abs_tol 
                    + rel_tol*cp.linalg.norm(kr_u_t @ theta)) 
        
        if (cp.linalg.norm(res_pri) <= eps_pri and 
            cp.linalg.norm(res_dual) <= eps_dual):
            converged = True
            break

    return u, z, theta, j + 1, converged, gcg_nonconverged

def kronecker_mvm(kr_u, vec, shape):
    """Apply a Kronecker-structured matrix-vector product.

    Args:
        kr_u (list): List of covariance matrices for local operator.
        vec (cupy.ndarray): Input vector.
        shape (tuple): shape of original tensor.
    
    Returns:
        cupy.ndarray: Vectorised result of the Kronecker matrix-vector product.
    """
    D = len(shape) 
    x = vec.reshape(shape, order = 'F') 

    for d in range(D-1, -1, -1): 
        x_unfold = unfold(x, d) 
        x_unfold = kr_u[d] @ x_unfold 
        x = fold(x_unfold, shape, d) 

    return x.ravel(order = 'F')

def local_operator(vec, pos_obs, kr_u, 
                   gamma, lambda_, N, 
                   total_data, buffer=None):
    """Constructs the local linear operator used in the local ADMM algorithm.

    Args:
        vec (cupy.ndarray): vector to be multiplied by the local linear operator.
        pos_obs (list): list of observed entries.
        kr_u (list): list of covariance matrices for the local linear operator.
        gamma (float): covariance regularisation parameter.
        lambda_ (float): local ADMM penalty parameter.
        N (int): number of total entries.

    Returns:
        cupy.ndarray: linear operator used in the Conjugate Gradient method for the local ADMM optimisation steps of the QKTF algorithm. 
    """
    buffer[:] = 0.0
    buffer[pos_obs] = vec 
    Ap = kronecker_mvm(kr_u, buffer, N) 
    return lambda_ * Ap[pos_obs] + ((gamma/total_data)*vec)

def local_admm(lambda_, gamma, w_prior, xi, x, Kr, pos_obs, total_data, YR_tilde,
               inner_maxiter, tau, cg_maxiter, rel_tol, abs_tol, verbose=False):
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
    N = numpy.array(YR_tilde.shape) 
    n_obs = pos_obs[0].shape[0] 
    Y_obs = (YR_tilde.ravel(order = 'F'))[pos_obs[0]]
    x0 = w_prior.copy() 

    w_full = cp.zeros(total_data, dtype=Y_obs.dtype) 

    def matvec(vec): 
        return local_operator(vec, pos_obs, Kr,
                              gamma, lambda_, N,
                              total_data, buffer=w_full)

    ar = linalg.LinearOperator((n_obs, n_obs), 
                               matvec=matvec, 
                               dtype=Y_obs.dtype)

    assert inner_maxiter > 0, "local_admm requires at least one ADMM sweep"

    # ----- ADMM iterations -----
    converged = False
    lcg_nonconverged = 0
    for j in range(inner_maxiter):
        x_prev = x.copy() 
        w_prev = x0.copy() 

        # ----- CG solve -----
        b = lambda_ * (Y_obs - x) + xi
        w, info = linalg.cg(ar, b, x0=x0, atol=1e-4, maxiter=cg_maxiter) 
        if info != 0:
            lcg_nonconverged += 1
        x0 = w
        w_full[:] = 0.0
        w_full[pos_obs] = w
        r = kronecker_mvm(Kr, w_full, N) 

        # ----- Dual update -----
        alpha = n_obs * lambda_ 
        x = prox_map(Y_obs - r[pos_obs] + (xi/lambda_), alpha, tau) 
    
        # ----- Lagrangian update -----
        xi = xi - lambda_ * (r[pos_obs] + x - Y_obs)

        # ----- Convergence checks -----
        res_pri = r[pos_obs] + x - Y_obs
        res_dual = lambda_*(x - x_prev)
        eps_pri = cp.sqrt(n_obs)*abs_tol 
        + rel_tol*cp.maximum(cp.maximum(cp.linalg.norm(r[pos_obs]), 
                                        cp.linalg.norm(x)), 
                                        cp.linalg.norm(Y_obs))
        eps_dual = cp.sqrt(total_data)*abs_tol 
        + rel_tol*cp.linalg.norm(xi)

        pri_ratio = float(cp.linalg.norm(res_pri)/eps_pri)
        dual_ratio = float(cp.linalg.norm(res_dual)/eps_dual)

        
        if (cp.linalg.norm(res_pri) <= eps_pri and 
            cp.linalg.norm(res_dual) <= eps_dual):
            converged = True
            break

    return r.ravel(order='F'), w, x, xi, j+1, converged, lcg_nonconverged

def qktf(I, Omega, 
         lengthscaleU: list, lengthscaleR: list, 
         varianceU: list, varianceR: list,
         tapering_range, d_MaternU, d_MaternR, R, 
         psi, sigma, gamma, lambda_, tau,
         max_iter, K0, epsilon, inner_maxiter, cg_maxiter=500,
         distance_matrix=None, verbose=False, seed=None):
    """Quantile Kernelised Tensor Factorisation (QKTF) algorithm for tensor completion.  

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
    N = I.shape 
    N = numpy.array(N)     
    D = I.ndim
    
    assert I.shape == Omega.shape 
    assert R > 0 
    assert 0 < tau < 1

    # ----- Pre-processing -----
    Omega = Omega.astype(bool) 
    pos_miss = cp.where(Omega == 0)
    num_obs = int(cp.sum(Omega).item()) 
    total_data = int(numpy.prod(N)) 

    mask_matrix = [unfold(Omega, d) for d in range(D)] 
    mask_matrixT = [mask_matrix[d].T for d in range(D)] 
    mask_flat = [mask_matrix[d].ravel(order='F') for d in range(D)] 
    pos_obs = [cp.where(mask_flat[d] == 1) for d in range(D)] 

    idx = cp.sum(mask_matrix[D-1], axis = 0) > 0 
    train_matrix = I[Omega] 
    centre = cp.mean(train_matrix)
    Isubmean = I - centre

    T = Isubmean * Omega 

    hyper_Ku, hyper_Kr = [None]*D, [None]*D 
    Ku, Kr = [None]*D, [None]*D 
    inv_Ku = [None]*D 

    for d in range(D-1): 
        if distance_matrix is not None and distance_matrix[d] is not None:
            a = distance_matrix[d]
        else:
            a = cp.arange(N[d]) 

        hyper_Ku[d] = [cp.log(lengthscaleU[d]), 
                       cp.log(varianceU[d])] 
        Ku[d] = cov_matern(d_MaternU, 
                           hyper_Ku[d], 
                           a) 
        inv_Ku[d] = cp.linalg.inv(Ku[d]) 

        hyper_Kr[d] = [cp.log(lengthscaleR[d]), 
                       cp.log(varianceR[d]), 
                       cp.log(tapering_range)]
        TaperM = bohman([hyper_Kr[d][2]], 
                        a)
        Kr[d] = csr_matrix(cov_matern(d_MaternR, 
                                      hyper_Kr[d][:2], 
                                      a)*TaperM) 

    inv_Ku[D-1] = cp.eye(N[D-1])
    Kr[D-1] = csr_matrix(eye(N[D-1])) 

    # ----- Initialisations -----
    X = T.copy() 
    X[pos_miss] = T.sum()/num_obs 

    rng = numpy.random.default_rng(seed)

    z, theta = [], []

    for d in range(D):
        dims = [N[i] for i in range(D) if i != d]
        unfold_shape = (int(cp.prod(cp.array(dims))), N[d])
        z.append(cp.zeros(unfold_shape))
        theta.append(cp.zeros(unfold_shape))

    U = [cp.asarray(rng.standard_normal((N[d], R))) for d in range(D)] 
    M = reconstruct_tensor(U, N) 
    Uvector = [U[d].ravel(order='F') for d in range(D)] 
    UTvector = [U[d].T.ravel(order='F') for d in range(D)] 

    Rtensor = cp.zeros(N) 
    w_warm = cp.zeros(num_obs)
    x = cp.zeros(num_obs) 
    xi = cp.zeros(num_obs) 
    Rvector = Rtensor.ravel(order='F') 
    Rvector_temp = Rtensor.ravel(order='F') 

    X[pos_miss] = M[pos_miss] + Rtensor[pos_miss] 

    d_all = cp.arange(D) 
    train_norm = cp.linalg.norm(T) 
    last_ten = T.copy() 
    pbar = tqdm(total=max_iter, desc="QKTF Iterations") 
    iter = 0 

    global_sweep_history = [] 
    global_hit_cap = 0
    global_cg_nonconverged = 0
    local_sweep_history = []
    local_hit_cap = 0
    local_cg_nonconverged = 0

    while True: 
        Gtensor = X - Rtensor 
        Gtensor_mask = Gtensor * Omega

        # Global component iteration
        global_sweeps = []
        l_sweeps, l_converged = None, None

        for d in range(D): 
            dsub = cp.delete(d_all, d) 
            dsub = cp.array(dsub) 
            Gtensor_unfold = unfold(Gtensor_mask, d).T 
            KrU = build_khatri_rao(U, dsub) 

            # Actual Global ADMM optimisation call.
            UTvector[d], z[d], theta[d], g_sweeps, g_converged, gcg_nonconverged = global_admm(
                inv_Ku[d], KrU, mask_matrixT[d], Gtensor_unfold,
                UTvector[d], z[d], theta[d], psi, sigma, inner_maxiter, tau, R,
                num_obs, cg_maxiter=cg_maxiter, rel_tol=1e-4, abs_tol=1e-4, verbose=verbose
                )
            U[d] = (UTvector[d].reshape(R, N[d], order = 'F')).T 
            global_sweep_history.append(g_sweeps)
            global_sweeps.append(g_sweeps)
            if not g_converged:
                global_hit_cap += 1
            global_cg_nonconverged += gcg_nonconverged
        
        M = reconstruct_tensor(U, N) 
        X[pos_miss] = M[pos_miss] + Rtensor[pos_miss] 
        
        if iter >= K0:
            Ltensor = X - M 
            Ltensor_mask = Ltensor * Omega 

            # Actual Local ADMM optimisation call.
            Rvector, w_warm, x, xi, l_sweeps, l_converged, lcg_nonconverged = local_admm(
                lambda_, gamma, w_warm, xi, x, Kr, pos_obs[0], total_data,
                Ltensor_mask, inner_maxiter, tau, cg_maxiter=cg_maxiter, rel_tol=1e-4, abs_tol=1e-4, verbose=verbose
                )

            local_sweep_history.append(l_sweeps)
            if not l_converged:
                local_hit_cap += 1
            local_cg_nonconverged += lcg_nonconverged

            Rtensor = Rvector.reshape(N, order = 'F')
        else:
            Rtensor[:] = 0.0 

        X[pos_miss] = M[pos_miss] + Rtensor[pos_miss]
        Xori = X + centre 

        # Convergence checks
        iter += 1 
        tol = cp.linalg.norm((X - last_ten))/train_norm 
        last_ten = X.copy() 

        pbar.update(1)
        pbar.set_postfix({
            'tol': f'{tol:.2e}',
            'g_sweeps (max)': int(max(global_sweeps)),
            'g_hit_cap': sum(1 for s in global_sweeps if s >= inner_maxiter),
            'l_sweeps': ('-' if l_sweeps is None else l_sweeps)
        })
        
        if cp.isnan(tol) or cp.isinf(tol): 
            pbar.set_postfix({'tol': f'{tol:.2e}', 'epoch': iter})
            break
        
        if (tol < epsilon and iter > K0) or (iter >= max_iter):
            pbar.close()
            if (iter >= max_iter):
                print("Maximum number of iterations reached.")
            break

    # ========== ADMM convergence summary ==========
    if len(global_sweep_history) > 0:
        g_arr = numpy.array(global_sweep_history)
        print(f"global_admm: {len(g_arr)} calls, sweeps used avg={g_arr.mean():.1f} "
              f"max={g_arr.max()} (cap={inner_maxiter}); hit cap in {global_hit_cap}/{len(g_arr)} "
              f"calls ({100 * global_hit_cap / len(g_arr):.1f}%) "
              f"{global_cg_nonconverged} inner CG solve(s) did not reach atol within cg_maxiter={cg_maxiter}")
        
    if len(local_sweep_history) > 0:
        l_arr = numpy.array(local_sweep_history)
        print(f"local_admm: {len(l_arr)} calls, sweeps used avg={l_arr.mean():.1f} "
              f"max={l_arr.max()} (cap={inner_maxiter}); hit cap in {local_hit_cap}/{len(l_arr)} "
              f"calls ({100 * local_hit_cap / len(l_arr):.1f}%) "
              f"{local_cg_nonconverged} inner CG solve(s) did not reach atol within cg_maxiter={cg_maxiter}")
        
    return Xori, Rtensor, M + centre