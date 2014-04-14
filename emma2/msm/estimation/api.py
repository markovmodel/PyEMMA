r"""
========================
Emma2 MSM Estimation API
========================

"""

__docformat__ = "restructuredtext en"

import numpy as np
import sparse.count_matrix
import sparse.connectivity
import sparse.likelihood
import sparse.transition_matrix
import dense.transition_matrix

from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import emma2.util.pystallone as stallone


__all__=['count_matrix',
         'cmatrix', 
         'connected_sets',
         'largest_connected_set',
         'largest_connected_submatrix',
         'is_connected',
         'transition_matrix',
         'log_likelihood',
         'tmatrix_sampler']

################################################################################
# Count matrix
################################################################################

# DONE: Benjamin 
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

# DONE: Benjamin 
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
    r"""Generate a countmatrix for the milestoning process on the
    given core sets.
    
    """
    raise NotImplementedError('Not implemented.')

# shortcut
cmatrix_cores=count_matrix_cores


################################################################################
# Connectivity
################################################################################

# DONE: Ben Implement in Python directly
def connected_sets(C, directed=True):
    r"""Connected components for a directed graph with edge-weights
    given by the count matrix.
    
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
    if isdense(C):
        # this should not be necessary because sparse.connectivity in principle works with dense matrices.
        # however there seems to be a bug for 2x2 matrices, therefore we use this detour
        return sparse.connectivity.connected_sets(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.connected_sets(C, directed=directed)

# DONE: Ben 
def largest_connected_set(C, directed=True):
    r"""Largest connected component for a directed graph with edge-weights
    given by the count matrix.
    
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
    if isdense(C):
        # this should not be necessary because sparse.connectivity in principle works with dense matrices.
        # however there seems to be a bug for 2x2 matrices, therefore we use this detour
        return sparse.connectivity.largest_connected_set(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.largest_connected_set(C, directed=directed)

# DONE: Ben 
def largest_connected_submatrix(C, directed=True):
    r"""Compute the count matrix on the largest connected set.
    
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
    if isdense(C):
        # this should not be necessary because sparse.connectivity in principle works with dense matrices.
        # however there seems to be a bug for 2x2 matrices, therefore we use this detour
        return sparse.connectivity.largest_connected_submatrix(csr_matrix(C), directed=directed).toarray()
    else:
        return sparse.connectivity.largest_connected_submatrix(C, directed=directed)

# shortcut
connected_cmatrix=largest_connected_submatrix
__all__.append('connected_cmatrix')

# DONE: Jan
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
    if isdense(C):
        # this should not be necessary because sparse.connectivity in principle works with dense matrices.
        # however there seems to be a bug for 2x2 matrices, therefore we use this detour
        return sparse.connectivity.is_connected(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.is_connected(C, directed=directed)

# TODO: Implement in Python directly
def mapping(set):
    """
    Constructs two dictionaries that map from the set values to their
    indexes, and vice versa.
    
    Parameters
    ----------
    set : array-like of integers 
        a set of selected states
    
    Returns
    -------
    dict : python dictionary mapping original to internal states 
    
    """   
    raise NotImplementedError('Not implemented.')


################################################################################
# Priors
################################################################################

# DONE: Frank
def prior_neighbor(Z, alpha = 0.001):
    """
    Returns a neighbor prior of strength alpha associated with the given count matrix.
    
    Prior is defined by 
        b_ij = alpha  if Z_ij+Z_ji > 0
        b_ij = 0      else
    
    Parameters
    ----------
    Z : numpy ndarray or scipy.sparse matrix (n,n)
        Count matrix
    alpha : float (default 0.001 counts)
    
    Returns
    -------
    B : numpy ndarray or scipy.sparse matrix (n,n), same type as Z
        Prior count matrix    
    """
    B = alpha * (Z + Z.transpose()).sign()
    return B

__all__.append('prior_neighbor')


# DONE: Frank
def prior_const(n, alpha = 0.001):
    """
    Returns a constant prior of strength alpha, i.e. b_ij=alpha for all i,j
    
    Parameters
    ----------
    n : the number of states. Determines the size of the return (n,n)
    alpha : float (default 0.001 counts)
    
    Returns
    -------
    B : numpy ndarray 
        Prior count matrix    
    """
    B = alpha*np.ones((n,n))
    return B

__all__.append('prior_const')


################################################################################
# Transition matrix
################################################################################

# DONE: Frank implemented dense (Nonreversible + reversible with fixed pi)
# DONE: Jan Implement in Python directly (Nonreversible)
# Done: Martin Map to Stallone (Reversible)
def transition_matrix(C, reversible=False, mu=None, **kwargs):
    """
    Estimate the transition matrix from the given countmatrix.
    
    The transition matrix is a maximum likelihood estimate (MLE)
    of the probability distribution of transition matrices
    with parameters given by the count matrix.
    
    Parameters
    ----------
    C : numpy ndarray or scipy.sparse matrix
        Count matrix
    reversible : bool (optional)
        If True restrict the ensemble of transition matrices
        to those having a detailed balance symmetry otherwise
        the likelihood optimization is carried out over the whole
        space of stochastic matrices.
    mu : array_like
        The stationary distribution of the MLE transition matrix.
    **kwargs: Optional algorithm-specific parameters. See below for special cases
    
    Returns
    -------
    P : numpy ndarray, shape=(n, n) or scipy.sparse matrix
       The MLE transition matrix. P has the same data type (dense or sparse) 
       as the input matrix C
    
    Reversible transition matrix estimation
    ---------------------------------------
    For reversible estimation, an iterative solver is used which accepts the
    following extra arguments and return behavior:
    Xinit = None : ndarray (n,n)
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) t, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    nmax = 1000000 : int
        maximum number of iterations before the method exits
    convtol = 1e-8 : float
        convergence tolerance. This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (x_i = sum_k x_ik). The relative stationary probability changes
        e_i = (x_i^(1) - x_i^(2))/(x_i^(1) + x_i^(2)) are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, |e_i|_2, is compared to convtol.
    return_statdist = False : Boolean
        If set to true, the stationary distribution is also returned
    return_conv = False : Boolean
        If set to true, the likelihood history and the pi_change history is returned.
    
    The estimator returns by default only P, but may also return
    (T,pi) or (T,lhist,pi_changes) or (T,pi,lhist,pi_changes) depending on the return settings
    T : ndarray (n,n)
        transition matrix. This is the only return for return_statdist = False, return_conv = False
    (pi) : ndarray (n)
        stationary distribution. Only returned if return_statdist = True
    (lhist) : ndarray (k)
        likelihood history. Has the length of the number of iterations needed. 
        Only returned if return_conv = True
    (pi_changes) : ndarray (k)
        history of likelihood history. Has the length of the number of iterations needed. 
        Only returned if return_conv = True
    
    """
    if (issparse(C)):
        sparse_mode = True
    elif (isdense(C)):
        sparse_mode = False
    else:
        raise NotImplementedError('C has an unknown type.')
    
    if reversible:
        if mu is None:
            try:
                if sparse_mode:
                    # currently no sparse impl, so we abuse dense impl (may be inefficient)
                    return csr_matrix(dense.transition_matrix.estimate_transition_matrix_reversible(C.toarray(),**kwargs))
                    ## Call to stallone. Currently deactivated because our impl is newer:
                    # Cs = stallone.ndarray_to_stallone_array(1.0*C.toarray())
                    ## T is of type stallone.IDoubleArray, so wrap it in an ndarray
                    # return csr_matrix(stallone.stallone_array_to_ndarray(stallone.API.msm.estimateTrev(Cs)))
                else:
                    return dense.transition_matrix.estimate_transition_matrix_reversible(C,**kwargs)
                    ## Call to stallone. Currently deactivated because our impl is newer:
                    # Cs = stallone.ndarray_to_stallone_array(1.0*C)
                    # T is of type stallone.IDoubleArray, so wrap it in an ndarray
                    # return stallone.stallone_array_to_ndarray(stallone.API.msm.estimateTrev(Cs))
            except stallone.JavaException as je:
                raise RuntimeError(je)
        else:
            if sparse_mode:
                # Sparse, reversible, fixed pi (currently using dense with sparse conversion)
                return csr_matrix(dense.transition_matrix.transition_matrix_reversible_fixpi(C.toarray(), mu))
            else:
                # Dense,  reversible, fixed pi
                return dense.transition_matrix.transition_matrix_reversible_fixpi(C, mu)
    else: # nonreversible estimation
        if mu is None:
            if sparse_mode:
                # Sparse,  nonreversible
                return sparse.transition_matrix.transition_matrix_non_reversible(C)
            else:
                # Dense,  nonreversible
                return dense.transition_matrix.transition_matrix_non_reversible(C)
        else:
            raise NotImplementedError('nonreversible mle with fixed stationary distribution not implemented.')

tmatrix = transition_matrix
__all__.append('tmatrix')

# DONE: FN+Jan 
def tmatrix_cov(C, k=None):
    r"""Nonreversible covariance matrix of transition matrix
    
    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    k : int (optional)
        If set, only the covariance matrix for this row is returned
       
    Returns
    -------
    cov : 
        
    """    
    return sparse.transition_matrix.tmatrix_cov(C, k)


# DONE: FN+Jan Implement in Python directly
def log_likelihood(C, T):
    r"""Log-likelihood of the count matrix given a transition matrix.

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    T : scipy.sparse matrix
        Transition matrix

    Returns
    -------
    logL : float
        Log-likelihood of the count matrix           
    
    """
    if issparse(C) and issparse(T):
        return sparse.likelihood.log_likelihood(C, T)
    else: 
        # use the dense likelihood calculator for all other cases
        # if a mix of dense/sparse C/T matrices is used, then both
        # will be converted to ndarrays.
        if (not isinstance(C, np.ndarray)):
            C = np.array(C)
        if (not isinstance(T, np.ndarray)):
            T = np.array(T)
        # computation is still efficient, because we only use terms
        # for nonzero elements of T
        nz = np.nonzero(T)
        return np.dot(C[nz], np.log(T[nz]))


# TODO: this function can be mixed dense/sparse, so maybe we should
# change the place for this function.
def error_perturbation(C, sensitivity):
    r"""Compute error perturbation.
    
    Parameters
    ----------
    C : count matrix 
    sensitivity : sensitivity matrix or tensor of
        size (m x n x n) where m is the dimension of the target
        quantity and (n x n) is the size of the transition matrix.
        The sensitivity matrix should be evaluated at an appropriate
        maximum likelihood or mean of the transition matrix estimated
        from C.

    Returns
    -------
    cov : (m x m) covariance matrix of the target quantity
    
    """
    return sparse.transition_matrix.error_perturbation(C, sensitivity)

# DONE: Martin Map to Stallone (Reversible)
def tmatrix_sampler(C, reversible=False, mu=None, P0=None):
    r"""Generate transition matrix sampler object.
    
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
    
    Returns
    -------
    sampler : A TransitionMatrixSampler object. In case reversible is True, 
        returns a stallone.ITransitionMatrixSampler instance.
    
    """
    if reversible:
        from emma2.util.pystallone import API as API, ndarray_to_stallone_array,\
            JavaException
        try:
            C = ndarray_to_stallone_array(C)
            if mu is not None:
                mu = ndarray_to_stallone_array(mu)
                sampler = API.msmNew.createTransionMatrixSamplerRev(C, mu)
            else:
                sampler = API.msmNew.createTransitionMatrixSamplerRev(C)
            return sampler
        except JavaException as je:
            raise RuntimeError(je)
    else:
        raise NotImplementedError('Non-reversible sampler not implemented.')


