import numpy as np
import scipy.linalg as scl
import scipy.sparse
from variational.solvers.direct import sort_by_norm
import msmtools.estimation as me

__all__ = ['bootstrapping_count_matrix', 'twostep_count_matrix', 'rank_decision',
           'oom_components', 'equilibrium_transition_matrix']


def bootstrapping_count_matrix(Ct, nbs=500):
    """
    Perform bootstrapping on trajectories to estimate uncertainties for singular values of count matrices.

    Parameters
    ----------
    Ct : csr-matrix
        count matrix of the data.

    nbs : int, optional, default=1000
        the number of re-samplings to be drawn from dtrajs

    Returns
    -------
    smean : ndarray(N,)
        mean values of singular values
    sdev : ndarray(N,)
        standard deviations of singular values
    """
    # Get the number of states:
    N = Ct.shape[0]
    # Get the number of transition pairs:
    T = Ct.sum()
    # Reshape and normalize the count matrix:
    p = Ct.toarray()
    p = np.reshape(p, (N*N,)).astype(np.float)
    p = p / T
    # Perform the bootstrapping:
    svals = np.zeros((nbs, N))
    for s in range(nbs):
        # Draw sample:
        sel = np.random.multinomial(T, p)
        # Compute the count-matrix:
        sC = np.reshape(sel, (N, N))
        # Compute singular values:
        svals[s, :] = scl.svdvals(sC)
    # Compute mean and uncertainties:
    smean = np.mean(svals, axis=0)
    sdev = np.std(svals, axis=0)

    return smean, sdev

def twostep_count_matrix(dtrajs, lag, N):
    """
    Compute all two-step count matrices from discrete trajectories.

    Parameters
    ----------
    dtrajs : list of discrete trajectories

    lag : int
        the lag time for count matrix estimation
    N : int
        the number of states in the discrete trajectories.

    Returns
    -------
    C2t : sparse csc-matrix (N, N, N)
        two-step count matrices for all states. C2t[:, n, :] is a count matrix for each n

    """
    # List all transition triples:
    rows = []
    cols = []
    states = []
    for dtraj in dtrajs:
        if dtraj.size > 2*lag:
            rows.append(dtraj[0:-2*lag])
            states.append(dtraj[lag:-lag])
            cols.append(dtraj[2*lag:])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    state = np.concatenate(states)
    data = np.ones(row.size)
    # Transform the rows and cols into a single list with N*+2 possible values:
    pair = N * row + col
    # Estimate sparse matrix:
    C2t = scipy.sparse.coo_matrix((data, (pair, state)), shape=(N*N, N))

    return C2t.tocsc()

def rank_decision(smean, sdev, tol=10.0):
    """
    Rank decision based on uncertainties of singular values.

    Parameters
    ----------
    smean : ndarray(N,)
        mean values of singular values for Ct
    sdev : ndarray(N,)
        standard errors of singular values for Ct
    tol : float, optional default=10.0
        accept singular values with signal-to-noise ratio >= tol_svd

    Returns
    -------
    ind : ndarray(N, dtype=bool)
        indicates which singular values are accepted.

    """
    # Determine signal-to-noise ratios of singular values:
    sratio = smean / sdev
    # Return Boolean array of accepted singular values:
    return sratio >= tol


def oom_components(Ct, C2t, rank_ind=None, lcc=None, tol_one=1e-2):
    """
    Compute OOM components and eigenvalues from count matrices:

    Parameters
    ----------
    Ct : ndarray(N, N)
        count matrix from data
    C2t : sparse csc-matrix (N, N, N)
        two-step count matrix from data for all states
    rank_ind : ndarray(N, dtype=bool), optional, default=None
        indicates which singular values are accepted. By default, all non-
        zero singular values are accepted.
    lcc : ndarray(N,)
        largest connected set of the count-matrix. Two step count matrix
        will be reduced to this set.
    tol_one : float, optiona, default=1e-2
        keep eigenvalues of absolute value less or equal 1+tol_one.

    Returns
    -------
    Xi : ndarray(M, N, M)
        matrix of set-observable operators
    omega: ndarray(M,)
        information state vector of OOM
    sigma : ndarray(M,)
        evaluator of OOM
    l : ndarray(M,)
        eigenvalues from OOM
    """
    # Decompose count matrix by SVD:
    if lcc is not None:
        Ct_svd = me.largest_connected_submatrix(Ct, lcc=lcc)
        N1 = Ct.shape[0]
    else:
        Ct_svd = Ct
    V, s, W = scl.svd(Ct_svd, full_matrices=False)
    # Make rank decision:
    if rank_ind is None:
        ind = (s >= np.finfo(float).eps)
    V = V[:, rank_ind]
    s = s[rank_ind]
    W = W[rank_ind, :].T

    # Compute transformations:
    F1 = np.dot(V, np.diag(s**-0.5))
    F2 = np.dot(W, np.diag(s**-0.5))

    # Apply the transformations to C2t:
    N = Ct_svd.shape[0]
    M = F1.shape[1]
    Xi = np.zeros((M, N, M))
    for n in range(N):
        if lcc is not None:
            C2t_n = C2t[:, lcc[n]].toarray()
            C2t_n = np.reshape(C2t_n, (N1, N1))
            C2t_n = me.largest_connected_submatrix(C2t_n, lcc=lcc)
        else:
            C2t_n = C2t[:, n].toarray()
            C2t_n = np.reshape(C2t_n, (N, N))
        Xi[:, n, :] = np.dot(F1.T, np.dot(C2t_n, F2))

    # Compute sigma:
    c = np.sum(Ct_svd, axis=1)
    sigma = np.dot(F1.T, c)
    # Compute eigenvalues:
    Xi_S = np.sum(Xi, axis=1)
    l, R = scl.eig(Xi_S.T)
    # Restrict eigenvalues to reasonable range:
    ind = np.where(np.logical_and(np.abs(l) <= (1+tol_one), np.real(l) >= 0.0))[0]
    l = l[ind]
    R = R[:, ind]
    # Sort and extract omega
    l, R = sort_by_norm(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l

def equilibrium_transition_matrix(Xi, omega, sigma, reversible=True, return_lcc=True):
    """
    Compute equilibrium transition matrix from OOM components:

    Parameters
    ----------
    Xi : ndarray(M, N, M)
        matrix of set-observable operators
    omega: ndarray(M,)
        information state vector of OOM
    sigma : ndarray(M,)
        evaluator of OOM
    reversible : bool, optional, default=True
        symmetrize corrected count matrix in order to obtain
        a reversible transition matrix.
    return_lcc: bool, optional, default=True
        return indices of largest connected set.

    Returns
    -------
    Tt_Eq : ndarray(N, N)
        equilibrium transition matrix
    lcc : ndarray(M,)
        the largest connected set of the transition matrix.
    """
    # Compute equilibrium transition matrix:
    Ct_Eq = np.einsum('j,jkl,lmn,n->km', omega, Xi, Xi, sigma)
    # Remove negative entries:
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    # Compute transition matrix after symmetrization:
    pi_r = np.sum(Ct_Eq, axis=0)
    if reversible:
        pi_c = np.sum(Ct_Eq, axis=1)
        pi_sym = pi_r + pi_c
        # Avoid zero row-sums. States with zero row-sums will be eliminated by active set update.
        ind0 = np.where(pi_sym == 0.0)[0]
        pi_sym[ind0] = 1.0
        Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi_sym[:, None]
    else:
        # Avoid zero row-sums. States with zero row-sums will be eliminated by active set update.
        ind0 = np.where(pi_r == 0.0)[0]
        pi_r[ind0] = 1.0
        Tt_Eq = Ct_Eq / pi_r[:, None]

    # Perform active set update:
    lcc = me.largest_connected_set(Tt_Eq)
    Tt_Eq = me.largest_connected_submatrix(Tt_Eq, lcc=lcc)

    if return_lcc:
        return Tt_Eq, lcc
    else:
        return Tt_Eq