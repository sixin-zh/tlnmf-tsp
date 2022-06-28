# QN-JD for real orthogonal basis
# License: MIT


from time import time

import numpy as np


def qndiag2(C, B0=None, weights=None, max_iter=1000, gtol=1e-6,
            lambda_min=1e-16, max_ls_tries=10, diag_only=False,
            return_B_list=False, verbose=0, cb_eval=None):
    """Joint diagonalization of matrices using the quasi-Newton method
    over orthogonal manifold/group

    Parameters
    ----------
    C : array-like, shape (n_samples, n_features, n_features)
        Set of matrices to be jointly diagonalized. C[0] is the first matrix,
        etc...

    B0 : None | array-like, shape (n_features, n_features)
        Initial point for the algorithm. If None, a whitener is used.

    weights : None | array-like, shape (n_samples,)
        Weights for each matrix in the loss:
        L = sum(weights * KL(C, C')) / sum(weights).
        No weighting (weights = 1) by default.

    max_iter : int, optional
        Maximum number of iterations to perform.

    gtol : float, optional
        A positive scalar giving the tolerance at which the
        algorithm is considered to have converged. The algorithm stops when
        |gradient| < tol = gtol*cst.

    lambda_min : float, optional
        A positive regularization scalar. Each eigenvalue of the Hessian
        approximation below lambda_min is set to lambda_min.

    max_ls_tries : int, optional
        Maximum number of line-search tries to perform.

    diag_only : bool, optional
        If true, the line search is done by computing only the diagonals of the
        dataset. The dataset is then computed after the line search.
        Taking diag_only = True might be faster than diag_only=False
        when the matrices are large (n_features > 200)

    return_B_list : bool, optional
        Chooses whether or not to return the list of iterates.

    verbose : integer, optional
        Prints informations about the state of the algorithm if True.

    Returns
    -------
    D : array-like, shape (n_samples, n_features, n_features)
        Set of matrices jointly diagonalized

    B : array, shape (n_features, n_features)
        Estimated joint diagonalizer matrix.

    infos : dict
        Dictionnary of monitoring informations, containing the times,
        gradient norms and objective values.

    References
    ----------
    P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm
    for joint diagonalization. Proc. ESANN 2019.
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
    https://hal.archives-ouvertes.fr/hal-01936887v1
    https://arxiv.org/abs/1811.11433
    """
    t0 = time()
    n_samples, n_features, _ = C.shape
    if B0 is None:
        C_mean = np.mean(C, axis=0)
        d, p = np.linalg.eigh(C_mean)
        B = p.T / np.sqrt(d[:, None])
    else:
        B = B0.copy()
    if weights is not None:  # normalize
        weights_ = weights / np.mean(weights)
    else:
        weights_ = None

    D = transform_set(B, C)
    current_loss = None

    # Monitoring
    if return_B_list:
        B_list = []
    t_list = []
    gradient_list = []
    loss_list = []
    gs_obj_dict = []
    gs_obj_iter = []
    if verbose:
        print('Running quasi-Newton for joint diagonalization')
        print(' | '.join([name.center(8) for name in
                         ["iter", "obj", "gradient"]]))

    n_jd = 0
    for t in range(max_iter):
        if cb_eval is not None:
            eval_dict = cb_eval(B,ite=n_jd)
            gs_obj_dict.append(eval_dict)
            gs_obj_iter.append(n_jd)
            
        if return_B_list:
            B_list.append(B.copy())
        t_list.append(time() - t0)
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        # Gradient
        G = np.average(D / diagonals[:, :, None], weights=weights_,
                       axis=0) - np.eye(n_features)
        g_norm = np.linalg.norm(G)
        if g_norm < gtol * np.sqrt(n_features):  # rescale by identity
            print('break with g_norm = ', g_norm, 'at t=', t)
            #print('B norm is',np.linalg.norm(B))
            break

        # Hessian coefficients
        h = np.average(diagonals[:, None, :] / diagonals[:, :, None],
                       weights=weights_, axis=0)
        D_anti = (G-G.T)/2
        h_symm = (h+h.T)/2
        # Quasi-Newton's direction
        det = h_symm - 1
        #print('min det', np.min(det))
        det[det < lambda_min] = lambda_min  # Regularize
        direction = -D_anti/det

        # project E to Tangent sp: a skew symmetrix matrix
        #direction = (direction - direction.T)/2

        # Line search with unitary projection of B
        success, new_D, new_B, new_loss, direction =\
                linesearch2(C, B, direction, current_loss, max_ls_tries, diag_only,
                            weights_)

        if success:
            D = new_D
            B = new_B
            current_loss = new_loss
            n_jd += 1
            #print('jd2: ite',n_jd,'current loss',current_loss)
        else:
            print('jd2: break with ls at t=',t)
            break
            #print('jd2: skip with ls at t=',t)
            #n_jd += 1


        # Monitoring
        gradient_list.append(g_norm)
        loss_list.append(current_loss)
        if verbose > 0 and t % verbose == 0:
            print(' | '.join([("%d" % (t + 1)).rjust(8),
                              ("%.2e" % current_loss).rjust(8),
                              ("%.2e" % g_norm).rjust(8)]))

    if cb_eval is not None:
        eval_dict = cb_eval(B,ite=n_jd)
        gs_obj_dict.append(eval_dict)
        gs_obj_iter.append(n_jd)
        
    infos = {'t_list': t_list, 'gradient_list': gradient_list,
             'loss_list': loss_list, 'gs_obj_dict':gs_obj_dict,
             'gs_obj_iter':gs_obj_iter}
    if return_B_list:
        infos['B_list'] = B_list
    
    return B, infos

def transform_set(M, D, diag_only=False):
    n, p, _ = D.shape
    # D: (n,p,p)
    # M: (p,p)
    # return: M D M' (n,p,p)
    if not diag_only:
        op = np.zeros((n, p, p))
        for i, d in enumerate(D):
            op[i] = M.dot(d.dot(M.T))
    else:
        assert(0)
        op = np.zeros((n, p))
        for i, d in enumerate(D):
            op[i] = np.sum(M * d.dot(M.T), axis=0)
    return op

def unitary_projection(M):
    ''' Projects M on the rotation manifold
    Parameters
    ----------
    M : array, shape (M, M)
        Input matrix
    '''
    s, u = np.linalg.eigh(np.dot(M, M.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), M)

def loss2(B, D, is_diag=False, weights=None):
    n, p = D.shape[:2]
    if not is_diag:
        diagonals = np.diagonal(D, axis1=1, axis2=2)
    else:
        diagonals = D
    logdet = 0 # -np.linalg.slogdet(B)[1]
    if weights is None:
        #print('min diag',np.min(diagonals))
        return logdet + 0.5 * np.sum(np.log(diagonals)) / n
    else:
        return logdet + 0.5 * np.sum(weights[:, None] * np.log(diagonals)) / n

def gradient(D, weights=None):
    n, p, _ = D.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    return np.average(D / diagonals[:, :, None], weights=weights, axis=0) - np.eye(p)

def linesearch2(C, B, direction, current_loss, n_ls_tries, diag_only, weights):
    n, p, _ = C.shape
    step = 1.
    if current_loss is None:
        D = transform_set(B, C)
        current_loss = loss2(B, D)
        #print('init loss',current_loss)
    for n in range(n_ls_tries):
        M = np.eye(p) + step * direction
        new_B = unitary_projection(np.dot(M, B))
        new_D = transform_set(new_B, C, diag_only=diag_only)
        new_loss = loss2(new_B, new_D, diag_only, weights)
        if new_loss < current_loss:
            success = True
            break
        step /= 2.
    else:
        success = False
    # Compute new value of D if only its diagonal was computed
    if diag_only:
        assert(0)
        new_D = transform_set(M, D, diag_only=False)
    return success, new_D, new_B, new_loss, step * direction
