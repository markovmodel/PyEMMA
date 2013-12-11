"""This module contains the api definitions for the estimation module"""

import sparse.count_matrix
import sparse.connectivity
import sparse.transition_matrix

from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

__all__=['count_matrix', 'cmatrix', 'connected_sets', 'largest_connected_set',\
             'connected_count_matrix', 'is_connected', 'transition_matrix', 'log_likelihood',\
             'tmatrix_sampler']

################################################################################
# Count matrix
################################################################################

# DONE: Benjamin Implement in Python directly
def count_matrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix from given list(s) of integers.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The count matrix at given lag in coordinate list format.
    
    """
    if type(dtraj) is list:
        return sparse.count_matrix.count_matrix_mult(dtraj, lag, sliding=sliding)
    else:
        return sparse.count_matrix.count_matrix(dtraj, lag, sliding=sliding)

# DONE: Benjamin Implement in Python directly
def cmatrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix in from given list(s) of integers.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The countmatrix at given lag in coordinate list format.
        
    """
    return count_matrix(dtraj, lag, sliding=sliding)

# TODO: Implement in Python directly
def count_matrix_cores(dtraj, cores, lag, sliding=True):
    r"""Generate a countmatrix for the milestoning process on the given core sets
    """
    raise NotImplementedError('Not implemented.')

cmatrix_cores=count_matrix_cores

################################################################################
# Connectivity
################################################################################

# DONE: Ben Implement in Python directly
def connected_sets(C, directed=True):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.       

    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in the
        corresponding connected component. The list is sorted
        according to the size of the individual components. The
        largest connected set is the first entry in the list, lcc=cc[0].

    """
    return sparse.connectivity.connected_sets(C)

# DONE: Ben Implement in Python directly
def largest_connected_set(C, directed=True):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.       

    Returns
    -------
    lcc : array of integers
        The largest connected component of the directed graph.

    """
    return sparse.connectivity.largest_connected_set(C)

# DONE: Ben Implement in Python directly
def connected_count_matrix(C, directed=True):
    r"""Compute the count matrix of the largest connected set.

    The input count matrix is used as a weight matrix for the
    construction of a directed graph. The largest connected set of the
    constructed graph is computed. Vertices belonging to the largest
    connected component are used to generate a completely connected
    subgraph. The weight matrix of the subgraph is the desired
    completely connected count matrix.
    
    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True.       

    Returns
    -------
    C_cc : scipy.sparse matrix
        Count matrix of largest completely 
        connected set of vertices (states)

    """
    return sparse.connectivity.connected_count_matrix(C)

connected_cmatrix=connected_count_matrix

__all__.append('connected_cmatrix')

def is_connected(C, directed=True):
    """Check if C is a countmatrix for a completely connected process.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True.       

    Returns
    -------
    is_connected: bool
        True if C is countmatrix for a completely connected process
        False otherwise.

    """
    return sparse.connectivity.is_connected(C)

# TODO: Implement in Python directly
def mapping(set):
    """
    Constructs two dictionaries that map from the set values to their
    indexes, and vice versa.
    
    Parameters
    ----------
    set : array-like of integers 

    Returns
    -------
    dict : python dictionary mapping original to internal states 
    dict : python dictionary mapping internal to original states 

    """   
    raise NotImplementedError('Not implemented.')

################################################################################
# Transition matrix
################################################################################

# DONE: Jan Implement in Python directly (Nonreversible)
# TODO: Implement in Python directly (Reversible with stat dist)
# Done: Martin Map to Stallone (Reversible)
def transition_matrix(C, reversible=False, mu=None, **kwargs):
    """
    Estimate the transition matrix from the given countmatrix.

    The transition matrix is a maximum likelihood estimate (MLE)
    of the probability distribution of transition matrices
    with parameters given by the count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    reversible : bool (optional)
        If True restrict the ensemble of transition matrices
        to those having a detailed balance symmetry otherwise
        the likelihood optimization is carried out over the whole
        space of stochastic matrices.
    mu : array_like
        The stationary distribution of the MLE transition matrix.
    **kwargs: Optional algorithm-specific parameters

    Returns
    -------
    P : numpy ndarray, shape=(n, n) or scipy.sparse matrix
       The MLE transition matrix

    """
    if reversible:
        if mu is None:
            from emma2.util.pystallone import stallone_available
            if stallone_available == False:
                raise RuntimeError('stallone not available and reversible \
                     only impled there')
            from emma2.util.pystallone import API as API, ndarray_to_stallone_array, \
                JavaError, stallone_array_to_ndarray
            try:
                C = ndarray_to_stallone_array(C)
                # T is of type stallone.IDoubleArray, so wrap it in an ndarray
                return stallone_array_to_ndarray(API.msm.estimateTrev(C))
            except JavaError as je:
                raise RuntimeError(je.getJavaException())
        else:
            raise NotImplementedError('mle with fixed stationary distribution not implemented.')
    else:
        return sparse.transition_matrix.transition_matrix_non_reversible(C)        
            

tmatrix = transition_matrix
__all__.append('tmatrix')

# TODO: Jan Implement in Python directly
# TODO: Is C posterior or prior counts? 
def tmatrix_cov(C, k=None):
    """
    Computes a nonreversible covariance matrix of transition matrix elments
    
    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    k : row index (optional). 
        If set, only the covariance matrix for this row is returned.
        
    """
    
    return sparse.transition_matrix.tmatrix_cov(C, k)
        
# DONE: Jan Implement in Python directly
def log_likelihood(C, T):
    """
        likelihood of C given T
    """
    sparse.likelihood.log_likelihood(C, T)
    
# TODO: this function can be mixed dense/sparse, so maybe we should change the place for this function.
def error_perturbation(C, sensitivity):
    """
        C: count matrix 
        sensitivity: sensitivity matrix or tensor of
        size (m x n x n) where m is the dimension of the target
        quantity and (n x n) is the size of the transition matrix.
        The sensitivity matrix should be evaluated at an appropriate
        maximum likelihood or mean of the transition matrix estimated
        from C.  returns: (m x m) covariance matrix of the target
        quantity
    """
    return sparse.perturbation.error_perturbation(C, sensitivity)

# Done: Martin Map to Stallone (Reversible)
def tmatrix_sampler(C, reversible=False, mu=None, P0=None):
    """
    Parameters
    ----------
    C : ndarray, shape=(n, n) or scipy.sparse matrix
        Count matrix
    reversible : bool
        If true sample from the ensemble of transition matrices
        restricted to those obeying a detailed balance condition,
        else draw from the whole ensemble of stochastic matrices.
    mu : array_like
        The stationary distribution of the transition matrix samples.
    P0 : ndarray, shape=(n, n) or scipy.sparse matrix
        Starting point of the MC chain of the sampling algorithm.
        Has to obey the required constraints.

    Returns :
    ---------
    sampler : A TransitionMatrixSampler object. In case reversible is True, 
        returns a stallone.ITransitionMatrixSampler instance.

    """
    if reversible:
        from emma2.util.pystallone import stallone_available
        if not stallone_available:
                raise RuntimeError('stallone not available and reversible \
                     only impled there')
        from emma2.util.pystallone import API as API, ndarray_to_stallone_array,\
            JavaError
        try:
            C = ndarray_to_stallone_array(C)
            if mu is not None:
                mu = ndarray_to_stallone_array(mu)
                sampler = API.msmNew.createTransionMatrixSamplerRev(C, mu)
            else:
                sampler = API.msmNew.createTransitionMatrixSamplerRev(C)
            return sampler
        except JavaError as je:
            raise RuntimeError(je.getJavaException())
    else:
        raise NotImplementedError('Non-reversible sampler not implemented.')


