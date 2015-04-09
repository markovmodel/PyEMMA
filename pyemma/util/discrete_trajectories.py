__author__ = 'noe'

from pyemma.util.types import ensure_dtraj_list as _ensure_dtraj_list

r"""This module implements IO and manipulation function for discrete trajectories

Discrete trajectories are generally ndarrays of type integer
We store them either as single column ascii files or as ndarrays of shape (n,) in binary .npy format.

.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""

import numpy as np

from pyemma.util.annotators import shortcut
from pyemma.util.types import ensure_dtraj_list as _ensure_dtraj_list

################################################################################
# ascii
################################################################################

@shortcut('read_dtraj')
def read_discrete_trajectory(filename):
    """Read discrete trajectory from ascii file.

    The ascii file containing a single column with integer entries is
    read into an array of integers.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.

    Returns
    -------
    dtraj : (M, ) ndarray
        Discrete state trajectory.

    """
    with open(filename, "r") as f:
        lines=f.read()
        dtraj=np.fromstring(lines, dtype=int, sep="\n")
        return dtraj

@shortcut('write_dtraj')
def write_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to ascii file.

    The discrete trajectory is written to a
    single column ascii file with integer entries

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.

    dtraj : array-like
        Discrete state trajectory.

    """
    dtraj=np.asarray(dtraj)
    with open(filename, 'w') as f:
        dtraj.tofile(f, sep='\n', format='%d')

################################################################################
# binary
################################################################################

@shortcut('load_dtraj')
def load_discrete_trajectory(filename):
    r"""Read discrete trajectory form binary file.

    The binary file is a one dimensional numpy array
    of integers stored in numpy .npy format.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.

    Returns
    -------
    dtraj : (M,) ndarray
        Discrete state trajectory.

    """
    dtraj=np.load(filename)
    return dtraj

@shortcut('save_dtraj')
def save_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to binary file.

    The discrete trajectory is stored as ndarray of integers
    in numpy .npy format.

    Parameters
    ----------
    filename : str
        The filename of the discrete state trajectory file.
        The filename can either contain the full or the
        relative path to the file.


    dtraj : array-like
        Discrete state trajectory.

    """
    dtraj=np.asarray(dtraj)
    np.save(filename, dtraj)


################################################################################
# simple statistics
################################################################################


@shortcut('histogram')
def count_states(dtrajs):
    r"""returns a histogram count

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories

    Returns
    -------
    count : ndarray((n), dtype=int)
        the number of occurrances of each state. n=max+1 where max is the largest state index found.
    """
    # format input
    dtrajs = _ensure_dtraj_list(dtrajs)
    # make bincounts for each input trajectory
    nmax = 0
    bcs = []
    for i in range(len(dtrajs)):
        bc = np.bincount(dtrajs[i])
        nmax = max(nmax, bc.shape[0])
        bcs.append(bc)
    # construct total bincount
    res = np.zeros((nmax),dtype=int)
    # add up individual bincounts
    for i in range(len(bcs)):
        res[:bcs[i].shape[0]] += bcs[i]
    return res

def visited_set(dtrajs):
    r"""returns the set of states that have at least one count

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories

    Returns
    -------
    vis : ndarray((n), dtype=int)
        the set of states that have at least one count.
    """
    hist = count_states(dtrajs)
    return np.argwhere(hist > 0)[:,0]

@shortcut('nstates')
def number_of_states(dtrajs, only_used = False):
    r"""returns the number of states in the given trajectories.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    only_used = False : boolean
        If False, will return max+1, where max is the largest index used.
        If True, will return the number of states that occur at least once.
    """
    dtrajs = _ensure_dtraj_list(dtrajs)
    if only_used:
        # only states with counts > 0 wanted. Make a bincount and count nonzeros
        bc = count_states(dtrajs)
        return np.count_nonzero(bc)
    else:
        # all states wanted, included nonpopulated ones. return max + 1
        imax = 0
        for dtraj in dtrajs:
            imax = max(imax, np.max(dtraj))
        return imax+1

################################################################################
# indexing
################################################################################


def index_states(dtrajs, subset = None):
    """Generates a trajectory/time indexes for the given list of states

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used

    Returns
    -------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurances of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.

    """
    # check input
    dtrajs = _ensure_dtraj_list(dtrajs)
    # select subset unless given
    n = number_of_states(dtrajs)
    if subset is None:
        subset = range(n)
    else:
        if np.max(subset) >= n:
            raise ValueError('Selected subset is not a subset of the states in dtrajs.')
    # histogram states
    hist = count_states(dtrajs)
    # efficient access to which state are accessible
    is_requested = np.ndarray((n), dtype=bool)
    is_requested[:] = False
    is_requested[subset] = True
    # efficient access to requested state indexes
    full2states = np.zeros((n), dtype=int)
    full2states[subset] = range(len(subset))
    # initialize results
    res    = np.ndarray((len(subset)), dtype=object)
    counts = np.zeros((len(subset)), dtype=int)
    for i,s in enumerate(subset):
        res[i] = np.zeros((hist[s],2), dtype=int)
    # walk through trajectories and remember requested state indexes
    for i,dtraj in enumerate(dtrajs):
        for t,s in enumerate(dtraj):
            if is_requested[s]:
                k = full2states[s]
                res[k][counts[k],0] = i
                res[k][counts[k],1] = t
                counts[k] += 1
    return res

################################################################################
# sampling from state indexes
################################################################################


def sample_indexes_by_sequence(indexes, sequence):
    """Samples trajectory/time indexes according to the given sequence of states

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    sequence : array of integers
        A sequence of discrete states. For each state, a trajectory/time index will be sampled at which dtrajs
        have an occurrences of this state

    Returns
    -------
    indexes : ndarray( (N, 2) )
        The sampled index sequence.
        Index array with a number of rows equal to N=len(sequence), with rows consisting of a tuple (i, t),
        where i is the index of the trajectory and t is the time index within the trajectory.

    """
    N = len(sequence)
    res = np.zeros((N,2), dtype=int)
    for t in range(N):
        s = sequence[t]
        i = np.random.randint(indexes[s].shape[0])
        res[t,:] = indexes[s][i,:]

    return res

def sample_indexes_by_state(indexes, nsample, subset=None, replace=True):
    """Samples trajectory/time indexes according to the given sequence of states

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    nsample : int
        Number of samples per state. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indexes are available for a state.
    subset : ndarray((n)), optional, default = None
        array of states to be indexed. By default all states in dtrajs will be used
    replace : boolean, optional
        Whether the sample is with or without replacement

    Returns
    -------
    indexes : list of ndarray( (N, 2) )
        List of the sampled indices by state.
        Each element is an index array with a number of rows equal to N=len(sequence), with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indexes)
    # define set of states to work on
    if subset is None:
        subset = range(n)

    # list of states
    res = np.ndarray((len(subset)), dtype=object)
    for i in range(len(subset)):
        # sample the following state
        s = subset[i]
        # how many indexes are available?
        m_available = indexes[s].shape[0]
        # do we have no indexes for this state? Then insert empty array.
        if (m_available == 0):
            res[i] = np.zeros((0,2), dtype=int)
        elif replace:
            I = np.random.choice(m_available, nsample, replace=True)
            res[i] = indexes[s][I,:]
        else:
            I = np.random.choice(m_available, min(m_available,nsample), replace=False)
            res[i] = indexes[s][I,:]

    return res

def sample_indexes_by_distribution(indexes, distributions, nsample):
    """Samples trajectory/time indexes according to the given probability distributions

    Parameters
    ----------
    indexes : list of ndarray( (N_i, 2) )
        For each state, all trajectory and time indexes where this state occurs.
        Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
        with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
        within the trajectory.
    distributions : list or array of ndarray ( (n) )
        m distributions over states. Each distribution must be of length n and must sum up to 1.0
    nsample : int
        Number of samples per distribution. If replace = False, the number of returned samples per state could be smaller
        if less than nsample indexes are available for a state.

    Returns
    -------
    indexes : length m list of ndarray( (nsample, 2) )
        List of the sampled indices by distribution.
        Each element is an index array with a number of rows equal to nsample, with rows consisting of a
        tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

    """
    # how many states in total?
    n = len(indexes)
    for dist in distributions:
        if len(dist) != n:
            raise('Size error: Distributions must all be of length n (number of states).')

    # list of states
    res = np.ndarray((len(distributions)), dtype=object)
    for i in range(len(distributions)):
        # sample states by distribution
        sequence = np.random.choice(n, size=nsample, p=distributions[i])
        res[i] = sample_indexes_by_sequence(indexes, sequence)
    #
    return res
