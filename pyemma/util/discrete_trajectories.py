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
        res[i] = np.zeros((hist[s],2))
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
# sampling
################################################################################
