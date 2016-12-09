import numpy as np
import scipy.linalg as scl
import scipy.sparse

def bootstrapping_count_matrix(dtrajs, lag, nstates, nbs=500):
    """
    Perform bootstrapping on trajectories to estimate uncertainties for singular values of count matrices.

    Parameters
    ----------
    dtrajs : list of discrete trajectories

    lag : int
        the lag time for count matrix estimation
    nstates : int
        the number of states in the discrete trajectories.
    nbs : int, optional, default=500
        the number of re-samplings to be drawn from dtrajs.

    Returns
    -------
    smean : ndarray(N,)
        mean values of singular values
    sdev : ndarray(N,)
        standard deviations of singular values
    Ct : ndarray(N, N)
        actual count matrix of the data.
    """
    # List all transition pairs. Note that we only use pairs up to step -lag:
    rows = []
    cols = []
    for dtraj in dtrajs:
        if dtraj.size > 2*lag:
            rows.append(dtraj[0:-2*lag])
            cols.append(dtraj[lag:-lag])
    # Perform bootstrapping:
    ntraj = len(rows)
    svals = np.array((nbs, nstates))
    for s in range(nbs):
        # Draw sample:
        sel = np.random.choice(ntraj, ntraj, replace=True)
        srows = rows[sel]
        scols = cols[sel]
        # Compute count matrix:
        srows = np.concatenate(srows)
        scols = np.concatenate(scols)
        sdata = np.ones(rows.size)
        sC = scipy.sparse.coo_matrix((sdata, (srows, scols)), shape=(nstates, nstates))
        sC = sC.toarray()
        # Compute singular values:
        svals[s, :] = scl.svdvals(sC)
    # Compute mean and uncertainties:
    smean = np.mean(svals, axis=0)
    sdev = np.std(svals, axis=0)
    # Compute the actual count matrix if needed:
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.ones(row.size)
    Ct = scipy.sparse.coo_matrix((data, (row, col)), shape=(nstates, nstates))
    Ct = Ct.toarray()

    return smean, sdev, Ct

def compute_twostep_count_matrix(dtrajs, lag, nstates):
    """
    Compute all two-step count matrices from discrete trajectories.

    Parameters
    ----------
    dtrajs : list of discrete trajectories

    lag : int
        the lag time for count matrix estimation
    nstates : int
        the number of states in the discrete trajectories.

    Returns
    -------
    C2t : sparse csc-matrix (N, N, N)
        two-step count matrices for all states. C2t[:, n, :] is a count matrix for each n.

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
    # Transform the rows and cols into a single list with N*+2 possible values:
    pair = nstates * row + col
    # Estimate sparse matrix:
    data = np.ones(pair.size)
    C2t = scipy.sparse.coo_matrix((data, (pair, state)), shape=(nstates*nstates, nstates))

    return C2t.tocsc()

