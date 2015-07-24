# -*- coding: utf-8 -*-

# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""
=========================
PyEMMA MSM Estimation API
=========================

"""

__docformat__ = "restructuredtext en"

import warnings

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import sparse.count_matrix
import sparse.connectivity
import sparse.likelihood
import sparse.transition_matrix
import sparse.prior
import sparse.mle_trev_given_pi
import sparse.mle_trev

import dense.bootstrapping
import dense.transition_matrix
import dense.covariance
import dense.mle_trev_given_pi
import dense.tmatrix_sampler

from pyemma.util.annotators import shortcut
from pyemma.util.discrete_trajectories import count_states as _count_states
from pyemma.util.discrete_trajectories import number_of_states as _number_of_states
from pyemma.util.types import ensure_dtraj_list as _ensure_dtraj_list

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Fabian Paul", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['bootstrap_trajectories',
           'bootstrap_counts',
           'count_matrix',
           'count_states',
           'connected_sets',
           'error_perturbation',
           'is_connected',
           'largest_connected_set',
           'largest_connected_submatrix',
           'log_likelihood',
           'number_of_states',
           'prior_const',
           'prior_neighbor',
           'prior_rev',
           'transition_matrix',
           'log_likelihood',
           'sample_tmatrix',
           'tmatrix_cov',
           'tmatrix_sampler']

################################################################################
# Basic counting
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
    return _count_states(dtrajs)


@shortcut('nstates')
def number_of_states(dtrajs, only_used=False):
    r"""returns the number of states in the given trajectories.

    Parameters
    ----------
    dtraj : array_like or list of array_like
        Discretized trajectory or list of discretized trajectories
    only_used = False : boolean
        If False, will return max+1, where max is the largest index used.
        If True, will return the number of states that occur at least once.
    """
    return _number_of_states(dtrajs, only_used=only_used)


################################################################################
# Count matrix
################################################################################

# DONE: Benjamin 
@shortcut('cmatrix')
def count_matrix(dtraj, lag, sliding=True, sparse_return=True, nstates=None):
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
    sparse_return : bool (optional)
        Whether to return a dense or a sparse matrix.
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates)
    
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
    
    >>> import numpy as np
    >>> from pyemma.msm.estimation import count_matrix

    >>> dtraj = np.array([0, 0, 1, 0, 1, 1, 0])
    >>> tau = 2
    
    Use the sliding approach first

    >>> C_sliding = count_matrix(dtraj, tau)

    The generated matrix is a sparse matrix in COO-format. For
    convenient printing we convert it to a dense ndarray.

    >>> C_sliding.toarray()
    array([[ 1.,  2.],
           [ 1.,  1.]])

    Let us compare to the count-matrix we obtain using the lag
    approach
    
    >>> C_lag = count_matrix(dtraj, tau, sliding=False)
    >>> C_lag.toarray()
    array([[ 0.,  1.],
           [ 1.,  1.]])
    
    """
    # convert dtraj input, if it contains out of nested python lists to 
    # a list of int ndarrays.
    dtraj = _ensure_dtraj_list(dtraj)
    return sparse.count_matrix.count_matrix_mult(dtraj, lag, sliding=sliding, sparse=sparse_return, nstates=nstates)


# # TODO: Implement in Python directly
# def count_matrix_cores(dtraj, cores, lag, sliding=True):
# r"""Generate a countmatrix for the milestoning process on the
#     given core sets.
#     
#     """
#     raise NotImplementedError('Not implemented.')
# 
# # shortcut
# cmatrix_cores=count_matrix_cores


################################################################################
# Bootstrapping data
################################################################################

def bootstrap_trajectories(trajs, correlation_length):
    r"""Generates a randomly resampled trajectory segments.
    
    Parameters
    ----------
    trajs : array-like or array-like of array-like
        single or multiple trajectories. Every trajectory is assumed to be
        a statistically independent realization. Note that this is often not true and 
        is a weakness with the present bootstrapping approach.
            
    correlation_length : int
        Correlation length (also known as the or statistical inefficiency) of the data.
        If set to < 1 or > L, where L is the longest trajectory length, the 
        bootstrapping will sample full trajectories.
        We suggest to select the largest implied timescale or relaxation timescale as a 
        conservative estimate of the correlation length. If this timescale is unknown, 
        it's suggested to use full trajectories (set timescale to < 1) or come up with 
        a rough estimate. For computing the error on specific observables, one may use 
        shorter timescales, because the relevant correlation length is the integral of 
        the autocorrelation function of the observables of interest [3]. The slowest 
        implied timescale is an upper bound for that correlation length, and therefore 
        a conservative estimate [4].

    Notes
    -----
    This function can be called multiple times in order to generate randomly
    resampled trajectory data. In order to compute error bars on your observable
    of interest, call this function to generate resampled trajectories, and 
    put them into your estimator. The standard deviation of such a sample of 
    the observable is a model for the standard error.

    Implements a moving block bootstrapping procedure [1] for generation of 
    randomly resampled count matrixes from discrete trajectories. The corrlation length
    determines the size of trajectory blocks that will remain contiguous. 
    For a single trajectory N with correlation length t_corr < N, 
    we will sample floor(N/t_corr) subtrajectories of length t_corr using starting time t. 
    t is a uniform random number in [0, N-t_corr-1]. 
    When multiple trajectories are available, N is the total number of timesteps
    over all trajectories, the algorithm will generate resampled data with a total number
    of N (or slightly larger) time steps. Each trajectory of length n_i has a probability 
    of n_i to be selected. Trajectories of length n_i <= t_corr are returned completely.
    For longer trajectories, segments of length t_corr are randomly generated.

    Note that like all error models for correlated time series data, Bootstrapping 
    just gives you a model for the error given a number of assumptions [2]. The most 
    critical decisions are: (1) is this approach meaningful at all (only if the 
    trajectories are statistically independent realizations), and (2) select
    an appropriate timescale of the correlation length (see below).
    Note that transition matrix sampling from the Dirichlet distribution is a 
    much better option from a theoretical point of view, but may also be 
    computationally more demanding.

    References
    ----------
    .. [1] H. R. Kuensch. The jackknife and the bootstrap for general
        stationary observations, Ann. Stat. 3, 1217-41 (1989).
    .. [2] B. Efron. Bootstrap methods: Another look at the jackknife.
        Ann. Statist. 7 1-26 (1979).
    .. [3] T.W. Anderson. The Statistical Analysis of Time Series
        Wiley, New York (1971).
    .. [4] F. Noe and F. Nueske: A variational approach to modeling
        slow processes in stochastic dynamical systems.  SIAM
        Multiscale Model. Simul., 11 . pp. 635-655 (2013)
        
    """
    return dense.bootstrapping.bootstrap_trajectories(trajs, correlation_length)


def bootstrap_counts(dtrajs, lagtime):
    r"""Generates a randomly resampled count matrix given the input coordinates.
    
    Parameters
    ----------
    dtrajs : array-like or array-like of array-like
        single or multiple discrete trajectories. Every trajectory is assumed to be
        a statistically independent realization. Note that this is often not true and 
        is a weakness with the present bootstrapping approach.
            
    lagtime : int
        the lag time at which the count matrix will be evaluated

    Notes
    -----
    This function can be called multiple times in order to generate randomly
    resampled realizations of count matrices. For each of these realizations 
    you can estimate a transition matrix, and from each of them computing the 
    observables of your interest. The standard deviation of such a sample of 
    the observable is a model for the standard error.
    
    The bootstrap will be generated by sampling N/lagtime counts at time
    tuples (t, t+lagtime), where t is uniformly sampled over all trajectory
    time frames in [0,n_i-lagtime]. Here, n_i is the length of trajectory i
    and N = sum_i n_i is the total number of frames.

    See also
    --------
    bootstrap_trajectories
    
    """
    dtrajs = _ensure_dtraj_list(dtrajs)
    return dense.bootstrapping.bootstrap_counts(dtrajs, lagtime)


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
    
    >>> import numpy as np
    >>> from pyemma.msm.estimation import connected_sets

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> cc_directed = connected_sets(C)
    >>> cc_directed
    [array([0, 1]), array([2])]

    >>> cc_undirected = connected_sets(C, directed=False)
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
    
    >>> import numpy as np
    >>> from pyemma.msm.estimation import largest_connected_set

    >>> C =  np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> lcc_directed = largest_connected_set(C)
    >>> lcc_directed
    array([0, 1])

    >>> lcc_undirected = largest_connected_set(C, directed=False)
    >>> lcc_undirected
    array([0, 1, 2])
    
    """
    if isdense(C):
        return sparse.connectivity.largest_connected_set(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.largest_connected_set(C, directed=directed)


# DONE: Ben
@shortcut('connected_cmatrix')
def largest_connected_submatrix(C, directed=True, lcc=None):
    r"""Compute the count matrix on the largest connected set.   
    
    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed or
       undirected graph. Default is True
    lcc : (M,) ndarray, optional
       The largest connected set             
       
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
    
    >>> import numpy as np
    >>> from pyemma.msm.estimation import largest_connected_submatrix

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])

    >>> C_cc_directed = largest_connected_submatrix(C)
    >>> C_cc_directed
    array([[10,  1],
           [ 2,  0]])

    >>> C_cc_undirected = largest_connected_submatrix(C, directed=False)
    >>> C_cc_undirected
    array([[10,  1,  0],
           [ 2,  0,  3],
           [ 0,  0,  4]])
           
    """
    if isdense(C):
        return sparse.connectivity.largest_connected_submatrix(csr_matrix(C), directed=directed, lcc=lcc).toarray()
    else:
        return sparse.connectivity.largest_connected_submatrix(C, directed=directed, lcc=lcc)


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
    
    >>> import numpy as np
    >>> from pyemma.msm.estimation import is_connected
    
    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
    >>> is_connected(C)
    False
    
    >>> is_connected(C, directed=False)
    True    
    
    """
    if isdense(C):
        return sparse.connectivity.is_connected(csr_matrix(C), directed=directed)
    else:
        return sparse.connectivity.is_connected(C, directed=directed)


################################################################################
# priors
################################################################################

# DONE: Frank, Ben
def prior_neighbor(C, alpha=0.001):
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

    >>> import numpy as np
    >>> from pyemma.msm.estimation import prior_neighbor
    
    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_neighbor(C)
    >>> B
    array([[ 0.001,  0.001,  0.   ],
           [ 0.001,  0.   ,  0.001],
           [ 0.   ,  0.001,  0.001]])
           
    """

    if isdense(C):
        B = sparse.prior.prior_neighbor(csr_matrix(C), alpha=alpha)
        return B.toarray()
    else:
        return sparse.prior.prior_neighbor(C, alpha=alpha)


# DONE: Frank, Ben
def prior_const(C, alpha=0.001):
    r"""Constant prior for given count matrix.
    
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

    Notes
    -----
    The prior is defined as 

    .. math:: \begin{array}{rl} b_{ij}= \alpha & \forall i, j \end{array}

    Examples
    --------

    >>> import numpy as np
    >>> from pyemma.msm.estimation import prior_const
    
    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_const(C)
    >>> B
    array([[ 0.001,  0.001,  0.001],
           [ 0.001,  0.001,  0.001],
           [ 0.001,  0.001,  0.001]])
           
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

    Notes
    -----
    The reversible prior is a matrix with -1 on the upper triangle.
    Adding this prior respects the fact that
    for a reversible transition matrix the degrees of freedom
    correspond essentially to the upper triangular part of the matrix.

    The prior is defined as

    .. math:: b_{ij} = \left \{ \begin{array}{rl}
                       \alpha & i \leq j \\
                       0      & \text{elsewhere}
                       \end{array} \right .
                       
    Examples
    --------

    >>> import numpy as np
    >>> from pyemma.msm.estimation import prior_rev

    >>> C = np.array([[10, 1, 0], [2, 0, 3], [0, 1, 4]])
    >>> B = prior_rev(C)
    >>> B
    array([[-1., -1., -1.],
           [ 0., -1., -1.],
           [ 0.,  0., -1.]])
           
    """
    if isdense(C):
        return sparse.prior.prior_rev(C, alpha=alpha)
    else:
        warnings.warn("Prior will be a dense matrix for sparse input")
        return sparse.prior.prior_rev(C, alpha=alpha)


    ################################################################################


# Transition matrix
################################################################################

@shortcut('tmatrix')
def transition_matrix(C, reversible=False, mu=None, method='auto', **kwargs):
    r"""Estimate the transition matrix from the given countmatrix.   
    
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
    method : string (one of 'auto', 'dense' and 'sparse', optional, default='auto')
        Select which implementation to use for the estimation. 'dense' always
        selects the dense implementation, 'sparse' always selects the sparse
        one. 'auto' selectes the most efficient implementation according to
        the sparsity structure of the matrix: if the occupation of the C
        matrix is less then one third, select sparse. Else select dense.
        The type of the T matrix returned always matches the type of the 
        C matrix, irrespective of the method that was used to compute it.
    **kwargs: Optional algorithm-specific parameters. See below for special cases
    Xinit : (M, M) ndarray 
        Optional parameter with reversible = True.
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) t, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    maxiter : 1000000 : int
        Optional parameter with reversible = True.
        maximum number of iterations before the method exits
    maxerr : 1e-8 : float
        Optional parameter with reversible = True.
        convergence tolerance for transition matrix estimation.
        This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.
    return_statdist : False : Boolean
        Optional parameter with reversible = True.
        If set to true, the stationary distribution is also returned
    return_conv : False : Boolean
        Optional parameter with reversible = True.
        If set to true, the likelihood history and the pi_change history is returned.
    
    Returns
    -------
    P : (M, M) ndarray or scipy.sparse matrix
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

    Notes
    -----
    The transition matrix is a maximum likelihood estimate (MLE) of
    the probability distribution of transition matrices with
    parameters given by the count matrix.

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
    .. [2] Bowman, G R, K A Beauchamp, G Boxer and V S Pande. 2009.
        Progress and challenges in the automated construction of Markov state models for full protein systems.
        J. Chem. Phys. 131: 124101

    Examples
    --------

    >>> import numpy as np
    >>> from pyemma.msm.estimation import transition_matrix

    >>> C = np.array([[10, 1, 1], [2, 0, 3], [0, 1, 4]])

    Non-reversible estimate

    >>> T_nrev = transition_matrix(C)
    >>> T_nrev
    array([[ 0.83333333,  0.08333333,  0.08333333],
           [ 0.4       ,  0.        ,  0.6       ],
           [ 0.        ,  0.2       ,  0.8       ]])

    Reversible estimate

    >>> T_rev = transition_matrix(C, reversible=True)
    >>> T_rev
    array([[ 0.83333333,  0.10385552,  0.06281115],
           [ 0.35074675,  0.        ,  0.64925325],
           [ 0.04925323,  0.15074676,  0.80000001]])

    Reversible estimate with given stationary vector

    >>> mu = np.array([0.7, 0.01, 0.29])
    >>> T_mu = transition_matrix(C, reversible=True, mu=mu)
    >>> T_mu
    array([[ 0.94771371,  0.00612645,  0.04615984],
           [ 0.42885157,  0.        ,  0.57114843],
           [ 0.11142031,  0.01969477,  0.86888491]])    
    
    """
    if issparse(C):
        sparse_input_type = True
    elif isdense(C):
        sparse_input_type = False
    else:
        raise NotImplementedError('C has an unknown type.')

    if method == 'dense':
        sparse_computation = False
    elif method == 'sparse':
        sparse_computation = True
    elif method == 'auto':
        # heuristically determine whether is't more efficient to do a dense of sparse computation
        if sparse_input_type:
            dof = C.getnnz()
        else:
            dof = np.count_nonzero(C)
        dimension = C.shape[0]
        if dimension*dimension < 3*dof:
            sparse_computation = False
        else:
            sparse_computation = True
    else:
        raise ValueError(('method="%s" is no valid choice. It should be one of'
                          '"dense", "sparse" or "auto".')%method)

    # convert input type
    if sparse_computation and not sparse_input_type:
        C = coo_matrix(C)
    if not sparse_computation and sparse_input_type:
        C = C.toarray()

    if reversible:
        if mu is None:
            if sparse_computation:
                T = sparse.mle_trev.mle_trev(C, **kwargs)
            else:
                T = dense.transition_matrix.estimate_transition_matrix_reversible(C, **kwargs)
        else:
            if sparse_computation:
                # Sparse, reversible, fixed pi (currently using dense with sparse conversion)
                T = sparse.mle_trev_given_pi.mle_trev_given_pi(C, mu, **kwargs)
            else:
                T = dense.mle_trev_given_pi.mle_trev_given_pi(C, mu, **kwargs)
    else:  # nonreversible estimation
        if mu is None:
            if sparse_computation:
                # Sparse,  nonreversible
                T = sparse.transition_matrix.transition_matrix_non_reversible(C)
            else:
                # Dense,  nonreversible
                T = dense.transition_matrix.transition_matrix_non_reversible(C)
        else:
            raise NotImplementedError('nonreversible mle with fixed stationary distribution not implemented.')

    # convert return type
    if sparse_computation and not sparse_input_type:
        return T.toarray()
    elif not sparse_computation and sparse_input_type:
        return csr_matrix(T)
    else:
        return T


# DONE: FN+Jan+Ben Implement in Python directly
def log_likelihood(C, T):
    r"""Log-likelihood of the count matrix given a transition matrix.

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    T : (M, M) ndarray orscipy.sparse matrix
        Transition matrix
        
    Returns
    -------
    logL : float
        Log-likelihood of the count matrix

    Notes
    -----
        
    The likelihood of a set of observed transition counts
    :math:`C=(c_{ij})` for a given matrix of transition counts
    :math:`T=(t_{ij})` is given by 

    .. math:: L(C|P)=\prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    The log-likelihood is given by

    .. math:: l(C|P)=\sum_{i,j=1}^{M}c_{ij} \log p_{ij}.

    The likelihood describes the probability of making an observation
    :math:`C` for a given model :math:`P`.

    Examples
    --------

    >>> import numpy as np
    >>> from pyemma.msm.estimation import log_likelihood

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

    >>> C = np.array([[58, 7, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL = log_likelihood(C, T)
    >>> logL
    -38.280803472508182    

    >>> C = np.array([[58, 20, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL = log_likelihood(C, T)
    >>> logL
    -68.214409681430766

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105     
        
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
def tmatrix_cov(C, k=None):
    r"""Covariance tensor for non-reversible transition matrix posterior.
    
    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    k : int (optional)
        Return only covariance matrix for entires in the k-th row of
        the transition matrix
       
    Returns
    -------
    cov : (M, M, M) ndarray
        Covariance tensor for transition matrix posterior

    Notes
    -----
    The posterior of non-reversible transition matrices is 

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    Each row in the transition matrix is distributed according to a
    Dirichlet distribution with parameters given by the observed
    transition counts :math:`c_{ij}`.

    The covariance tensor
    :math:`\text{cov}[p_{ij},p_{kl}]=\Sigma_{i,j,k,l}` is zero
    whenever :math:`i \neq k` so that only :math:`\Sigma_{i,j,i,l}` is
    returned.
        
    """
    if issparse(C):
        warnings.warn("Covariance matrix will be dense for sparse input")
        C = C.toarray()
    return dense.covariance.tmatrix_cov(C, row=k)


# DONE: Ben
def error_perturbation(C, S):
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

    Notes
    -----

    **Scalar observable**

    The sensitivity matrix :math:`S=(s_{ij})` of a scalar observable
    :math:`f(T)` is defined as

    .. math:: S= \left(\left. \frac{\partial f(T)}{\partial t_{ij}} \right \rvert_{T_0} \right)

    evaluated at a suitable transition matrix :math:`T_0`.

    The sensitivity is the variance of the observable 

    .. math:: \mathbb{V}(f)=\sum_{i,j,k,l} s_{ij} \text{cov}[t_{ij}, t_{kl}] s_{kl}

    **Vector valued observable**

    The sensitivity tensor :math:`S=(s_{ijk})` for a vector
    valued observable :math:`(f_1(T),\dots,f_K(T))` is defined as

    .. math:: S= \left( \left. \frac{\partial f_i(T)}{\partial t_{jk}} \right\rvert_{T_0} \right)
    evaluated at a suitable transition matrix :math:`T_0`.

    The sensitivity is the covariance matrix for the observable 

    .. math:: \text{cov}[f_{\alpha}(T),f_{\beta}(T)]=\sum_{i,j,k,l} s_{\alpha i j} \text{cov}[t_{ij}, t_{kl}] s_{\beta kl}
    
    """

    if issparse(C):
        warnings.warn("Error-perturbation will be dense for sparse input")
        C = C.toarray()
    return dense.covariance.error_perturbation(C, S)


def _showSparseConversionWarning():
    warnings.warn('Converting input to dense, since method is '
                  'currently only implemented for dense matrices.', UserWarning)


def sample_tmatrix(C, nsample=1, reversible=False, mu=None, T0=None):
    r"""samples transition matrices from the posterior distribution

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    nsample : int
        number of samples to be drawn
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
    P : ndarray(n,n) or array of ndarray(n,n)
        sampled transition matrix (or multiple matrices if nsample > 1)

    Notes
    -----
    The transition matrix sampler generates transition matrices from
    the posterior distribution. The posterior distribution is given as
    a product of Dirichlet distributions

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    """
    if issparse(C):
        _showSparseConversionWarning()
        C = C.toarray()

    if reversible:
        raise NotImplementedError('Reversible transition matrix sampling is currently disabled')

    from dense.tmatrix_sampler import sample_nonrev
    return sample_nonrev(C, nsample=nsample)


def tmatrix_sampler(C, reversible=False, mu=None, T0=None):
    r"""Generate transition matrix sampler object.
    
    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
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
    sampler : A :py:class:dense.tmatrix_sampler.TransitionMatrixSampler object that can be used to generate samples.

    Notes
    -----
    The transition matrix sampler generates transition matrices from
    the posterior distribution. The posterior distribution is given as
    a product of Dirichlet distributions

    .. math:: \mathbb{P}(T|C) \propto \prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    The method can generate samples from the posterior under the follwing two constraints
    
    """
    # TODO: re-include reversible documention when implemented
    # The method can generate samples from the posterior under the follwing two constraints
    #
    # **Reversible sampling**
    #
    # Using a MCMC sampler outlined in .. [1] it is ensured that samples
    # from the posterior are reversible, i.e. there is a probability
    # vector :math:`(\mu_i)` such that :math:`\mu_i t_{ij} = \mu_j
    # t_{ji}` holds for all :math:`i,j`.
    #
    # **Reversible sampling with fixed stationary vector**
    #
    # Using a MCMC sampler outlined in .. [2] it is ensured that samples
    # from the posterior fulfill detailed balance with respect to a given
    # probability vector :math:`(\mu_i)`.
    #
    # References
    # ----------
    # .. [1] Noe, F. 2008. Probability distributions of molecular observables
    #     computed from Markov state models. J Chem Phys 128: 244103.
    # .. [2] Trendelkamp-Schroer, B and F Noe. 2013. Efficient Bayesian estimation
    #     of Markov model transition matrices with given stationary distribution.
    #     J Chem Phys 138: 164113.

    if issparse(C):
        _showSparseConversionWarning()
        C = C.toarray()

    if reversible:
        raise NotImplementedError('Reversible transition matrix sampling is currently disabled')

    from dense.tmatrix_sampler import TransitionMatrixSamplerNonrev
    sampler = TransitionMatrixSamplerNonrev(C)
    return sampler
