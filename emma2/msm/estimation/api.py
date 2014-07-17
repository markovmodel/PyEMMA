r"""
========================
Emma2 MSM Estimation API
========================

"""

__docformat__ = "restructuredtext en"

import warnings

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import sparse.count_matrix
import sparse.connectivity
import sparse.likelihood
import sparse.transition_matrix
import sparse.prior

import dense.transition_matrix
import dense.covariance

import emma2.util.pystallone as stallone
from emma2.util.log import getLogger
from emma2.msm.estimation.dense.tmatrix_sampler_jwrapper import ITransitionMatrixSampler

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__=['count_matrix',
         'cmatrix', 
         'connected_sets',
         'error_perturbation',
         'largest_connected_set',
         'largest_connected_submatrix',
         'is_connected',
         'prior_neighbor',
         'prior_const',
         'prior_rev',
         'transition_matrix',
         'tmatrix_cov',
         'log_likelihood',
         'tmatrix_sampler']

################################################################################
# Count matrix
################################################################################

# DONE: Benjamin 
def count_matrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix from given microstate trajectory.
    
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

    Notes
    -----
    Transition counts can be obtained from microstate trajectory using
    two methods. Couning at lag and slidingwindow counting.

    **Lag**
    
    This approach will skip all points in the trajectory that are
    seperated form the last point by less than the given lagtime
    :math:`\tau`.

    Transition counts :math:`c_{ij}(\tau)` are generated according to

    .. math:: c_{ij}(\tau)=\sum_{k=0}^{\left \lfloor \frac{N}{\tau} \right \rfloor -2}\chi_{i}(X_{k\tau})\chi_{j}(X_{(k+1)\tau}).

    :math:`\chi_{i}(x)` is the indicator function of :math:`i`, i.e
    :math:`\chi_{i}(x)=1` for :math:`x=i` and :math:`\chi_{i}(x)=0` for
    :math:`x \neq i`.

    **Sliding**

    The sliding approach slides along the trajectory and counts all
    transitions sperated by the lagtime :math:`\tau`.

    Transition counts :math:`c_{ij}(\tau)` are generated according to

    .. math:: c_{ij}(\tau)=\sum_{k=0}^{N-\tau-1} \chi_{i}(X_{k}) \chi_{j}(X_{k+\tau}).

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105

    Examples
    --------
    
    >>> from emma2.msm.estimation import count_matrix

    >>> dtraj=np.array([0, 0, 1, 0, 1, 1, 0])
    >>> tau=2
    
    Use the sliding approach first

    >>> C_sliding=count_matrix(dtraj, tau)

    The generated matrix is a sparse matrix in COO-format. For
    convenient printing we convert it to a dense ndarray.

    >>> C_sliding.toarray()
    array([[ 1.,  2.],
           [ 1.,  1.]])

    Let us compare to the count-matrix we obtain using the lag
    approach
    
    >>> C_lag=count_matrix(dtraj, tau, sliding=False)
    >>> C_lag.toarray()
    array([[ 0.,  1.],
           [ 1.,  1.]])
    
    """
    if type(dtraj) is list:
        return sparse.count_matrix.count_matrix_mult(dtraj, lag, sliding=sliding)
    else:
        return sparse.count_matrix.count_matrix(dtraj, lag, sliding=sliding)

# DONE: Benjamin 
def cmatrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix in from given microstate trajectory.
    
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

    See also
    --------
    count_matrix
    
    Notes
    -----
    This is a shortcut for a call to count_matrix.
        
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
    r"""Compute connected sets of microstates.

    Connected components for a directed graph with edge-weights
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
    
    Notes
    -----    
    Viewing the count matrix as the adjacency matrix of a (directed) graph
    the connected components are given by the connected components of that
    graph. Connected components of a graph can be efficiently computed
    using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------
    
    >>> from emma2.msm.estimation import connected_sets

    >>> C=np.array([10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> cc_directed=connected_sets(C)
    >>> cc_directed
    [array([0, 1]), array([2])]

    >>> cc_undirected=connected_sets(C, directed=False)
    >>> cc_undirected
    [array([0, 1, 2])]
    
    """
    if isdense(C):
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

    See also
    --------
    connected_sets

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed)
    graph the largest connected set is the largest connected set of
    nodes of the corresponding graph. The largest connected set of a graph
    can be efficiently computed using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------
    
    >>> from emma2.msm.estimation import largest_connected_set

    >>> C=np.array([10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> lcc_directed=largest_connected_set(C)
    >>> lcc_directed
    array([0, 1])

    >>> lcc_undirected=largest_connected_set(C, directed=False)
    >>> lcc_undirected
    array([0, 1, 2])
    
    """
    if isdense(C):
        return sparse.connectivity.largest_connected_set(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.largest_connected_set(C, directed=directed)

# DONE: Ben 
def largest_connected_submatrix(C, directed=True):
    r"""Compute the count matrix on the largest connected set.   
    
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
        
    See also
    --------
    largest_connected_set

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed)
    graph the larest connected submatrix is the adjacency matrix of
    the largest connected set of the corresponding graph. The largest
    connected submatrix can be efficiently computed using Tarjan's algorithm.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Examples
    --------
    
    >>> from emma2.msm.estimation import largest_connected_submatrix

    >>> C=np.array([10, 1, 0], [2, 0, 3], [0, 0, 4]])

    >>> C_cc_directed=largest_connected_submatrix(C)
    >>> C_cc_directed
    array([[10,  1],
           [ 2,  0]])

    >>> C_cc_undirected=largest_connected_submatrix(C, directed=False)
    >>> C_cc_undirected
    array([[10,  1,  0],
           [ 2,  0,  3],
           [ 0,  0,  4]])
           
    """
    if isdense(C):
        return sparse.connectivity.largest_connected_submatrix(csr_matrix(C), directed=directed).toarray()
    else:
        return sparse.connectivity.largest_connected_submatrix(C, directed=directed)

connected_cmatrix=largest_connected_submatrix
__all__.append('connected_cmatrix')

# DONE: Jan
def is_connected(C, directed=True):
    """Check connectivity of the given matrix.
    
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
        True if C is connected, False otherwise.
        
    See also
    --------
    largest_connected_submatrix
    
    Notes
    -----
    A count matrix is connected if the graph having the count matrix
    as adjacency matrix has a single connected component. Connectivity
    of a graph can be efficiently checked using Tarjan's algorithm.
    
    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.
        
    Examples
    --------
    
    >>> from emma2.msm.estimation import is_connected
    
    >>> C=np.array([10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> is_connected(C)
    False
    
    >>> is_connected(C)
    True    
    
    """
    if isdense(C):
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
# priors
################################################################################

# DONE: Frank, Ben
def prior_neighbor(C, alpha = 0.001):
    r"""Neighbor prior for the given count matrix.    
    
    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts
    
    Returns
    -------
    B : (M, M) ndarray or scipy.sparse matrix
        Prior count matrix

    Notes
    ------
    The neighbor prior :math:`b_{ij}` is defined as

    .. math:: b_{ij}=\left \{ \begin{array}{rl} 
                     \alpha & c_{ij}+c_{ji}>0 \\
                     0      & \text{else}
                     \end{array} \right .

    Examples
    --------
    >>> from emma2.msm.estimation import prior_neighbor
    
    >>> C=np.array([10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B=prior_neighbor(C)
    >>> B
    array([[ 0.001,  0.001,  0.   ],
           [ 0.001,  0.   ,  0.001],
           [ 0.   ,  0.001,  0.001]])
           
    """

    if isdense(C):
        B=sparse.prior.prior_neighbor(csr_matrix(C), alpha=alpha)
        return B.toarray()
    else:
        return sparse.prior.prior_neighbor(C, alpha=alpha)

# DONE: Frank, Ben
def prior_const(C, alpha = 0.001):
    """Constant prior of strength alpha.

    Prior is defined via

        b_ij=alpha for all i,j
    
    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts    
    
    Returns
    -------
    B : (M, M) ndarray 
        Prior count matrix    
        
    """
    if isdense(C):
        return sparse.prior.prior_const(C, alpha=alpha)
    else:
        warnings.warn("Prior will be a dense matrix for sparse input")
        return sparse.prior.prior_const(C, alpha=alpha)

__all__.append('prior_const')

# DONE: Ben
def prior_rev(C, alpha=-1.0):
    r"""Prior counts for sampling of reversible transition
    matrices.  

    Prior is defined as 

    b_ij= alpha if i<=j
    b_ij=0         else

    The reversible prior adds -1 to the upper triagular part of
    the given count matrix. This prior respects the fact that
    for a reversible transition matrix the degrees of freedom
    correspond essentially to the upper, respectively the lower
    triangular part of the matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    alpha : float (optional)
        Value of prior counts
       
    Returns
    -------
    B : (M, M) ndarray
        Matrix of prior counts        
    
    """
    if isdense(C):
        return sparse.prior.prior_rev(C, alpha=alpha)
    else:
        warnings.warn("Prior will be a dense matrix for sparse input")
        return sparse.prior.prior_rev(C, alpha=alpha)     


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
    Xinit = None : ndarray (n,n)
        Optional parameter with reversible = True.
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) t, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    maxiter = 1000000 : int
        Optional parameter with reversible = True.
        maximum number of iterations before the method exits
    maxerr = 1e-8 : float
        Optional parameter with reversible = True.
        convergence tolerance. This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (x_i = sum_k x_ik). The relative stationary probability changes
        e_i = (x_i^(1) - x_i^(2))/(x_i^(1) + x_i^(2)) are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, |e_i|_2, is compared to convtol.
    return_statdist = False : Boolean
        Optional parameter with reversible = True.
        If set to true, the stationary distribution is also returned
    return_conv = False : Boolean
        Optional parameter with reversible = True.
        If set to true, the likelihood history and the pi_change history is returned.
    
    Returns
    -------
    P : numpy ndarray, shape=(n, n) or scipy.sparse matrix
       The MLE transition matrix. P has the same data type (dense or sparse) 
       as the input matrix C.
    The reversible estimator returns by default only P, but may also return
    (P,pi) or (P,lhist,pi_changes) or (P,pi,lhist,pi_changes) depending on the return settings
    P : ndarray (n,n)
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
                log = getLogger()
                log.exception('Error during reversible tmatrix estimatation', je)
                raise
        else:
            if sparse_mode:
                # Sparse, reversible, fixed pi (currently using dense with sparse conversion)
                return csr_matrix(dense.transition_matrix.transition_matrix_reversible_fixpi(C.toarray(), mu,**kwargs))
            else:
                # Dense,  reversible, fixed pi
                return dense.transition_matrix.transition_matrix_reversible_fixpi(C, mu,**kwargs)
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

# DONE: Ben 
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
    if issparse(C):
        warnings.warn("Covariance matrix will be dense for sparse input")
        C=C.toarray()
    return dense.covariance.tmatrix_cov(C, row=k)

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


# DONE: Ben
def error_perturbation(C, sensitivity):
    r"""Error perturbation for given sensitivity matrix.

    Parameters
    ----------
    C : (M, M) ndarray
        Count matrix
    S : (M, M) ndarray or (K, M, M) ndarray
        Sensitivity matrix (for scalar observable) or sensitivity
        tensor for vector observable
        
    Returns
    -------
    X : float or (K, K) ndarray
        error-perturbation (for scalar observables) or covariance matrix
        (for vector-valued observable)
        
    """

    if issparse(C):
        warnings.warn("Error-perturbation will be dense for sparse input")
        C=C.toarray()
    return dense.covariance.error_perturbation(C, sensitivity)

def _showSparseConversionWarning():
    warnings.warn('Converting input to dense, since method is '
                  'currently only implemented for dense matrices.', UserWarning)

# DONE: Martin Map to Stallone (Reversible)
def tmatrix_sampler(C, reversible=False, mu=None, T0=None):
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
    T0 : ndarray, shape=(n, n) or scipy.sparse matrix
        Starting point of the MC chain of the sampling algorithm.
        Has to obey the required constraints.
    
    Returns
    -------
    sampler : A :py:class:dense.ITransitionMatrixSampler object.
    
    """
    if issparse(C):
        _showSparseConversionWarning()
        C=C.toarray()
    
    from emma2.util.pystallone import JavaException
    try:
        return ITransitionMatrixSampler(C, mu, reversible, Tinit=T0)
    except JavaException as je:
        log = getLogger()
        log.exception("Error during tmatrix sampling")
        raise
