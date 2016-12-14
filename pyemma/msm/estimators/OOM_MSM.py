import numpy as np
import scipy.linalg as scl
import scipy.sparse
from variational.solvers.direct import sort_by_norm
import msmtools.estimation as me


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

def oom_components(Ct, C2t, smean, sdev, tol=10.0, lcc=None):
    """
    Compute OOM components and eigenvalues from count matrices:

    Parameters
    ----------
    Ct : ndarray(N, N)
        count matrix from data
    C2t : sparse csc-matrix (N, N, N)
        two-step count matrix from data for all states
    smean : ndarray(N,)
        mean values of singular values for Ct
    sdev : ndarray(N,)
        standard errors of singular values for Ct
    tol : float, optional default
        accept singular values with signal-to-noise ratio >= tol
    lcc : ndarray(N,)
        largest connected set of the count-matrix. Two step count matrix
        will be reduced to this set.

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
    # Determine signal-to-noise ratios of singular values:
    sratio = smean / sdev
    # Decompose count matrix by SVD:
    if lcc is not None:
        Ct_svd = me.largest_connected_submatrix(Ct, lcc=lcc)
        N1 = Ct.shape[0]
        print N1
    else:
        Ct_svd = Ct
    V, s, W = scl.svd(Ct_svd, full_matrices=False)
    # Make rank decision:
    ind = np.where(sratio >= tol)[0]
    V = V[:, ind]
    s = s[ind]
    W = W[ind, :].T

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
    # Compute omega and all eigenvalues:
    Xi_S = np.sum(Xi, axis=1)
    l, R = scl.eig(Xi_S.T)
    l, R = sort_by_norm(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l

def equilibrium_transition_matrix(Xi, omega, sigma):
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

    Returns
    -------
    Tt_Eq : ndarray(N, N)
        equilibrium transition matrix
    """
    # Compute equilibrium transition matrix:
    Ct_Eq = np.einsum('j,jkl,lmn,n->km', omega, Xi, Xi, sigma)
    # Remove negative entries:
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    # Compute transition matrix after symmetrization:
    pi_r = np.sum(Ct_Eq, axis=0)
    pi_c = np.sum(Ct_Eq, axis=1)
    pi_sym = np.dot(np.diag(pi_r + pi_c), np.ones(Ct_Eq.shape))
    Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi_sym

    return Tt_Eq