r"""

======================
Emma2 MSM Analysis API
======================

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin.trendelkampschroer@gmail.com>

"""

__docformat__ = "restructuredtext en"

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import dense.assessment
import dense.committor
import dense.correlations
import dense.decomposition
import dense.expectations
import dense.pcca
import dense.sensitivity
import dense.mean_first_passage_time

import sparse.assessment
import sparse.decomposition
import sparse.expectations
import sparse.mean_first_passage_time

import emma2.msm.estimation as msmest

__author__=[]

__all__=['is_transition_matrix',
         'is_rate_matrix',
         'is_ergodic',
         'is_reversible',
         'stationary_distribution',
         'eigenvalues',
         'timescales',
         'eigenvectors',
         'rdl_decomposition',
         'expectation',
         'expected_counts',
         'expected_counts_stationary',
         'mfpt',
         'committor',
         'tpt',
         'pcca',
         'fingerprint_correlation',
         'fingerprint_relaxation',
         'evaluate_fingerprint',
         'correlation',
         'relaxation',
         'stationary_distribution_sensitivity',
         'eigenvalue_sensitivity',
         'timescale_sensitivity',
         'eigenvector_sensitivity',
         'mfpt_sensitivity',
         'committor_sensitivity']
# shortcuts added later:
# ['statdist', 'is_tmatrix', 'statdist_sensitivity']

_type_not_supported = \
    TypeError("given matrix is not a numpy.ndarray or a scipy.sparse matrix.")

################################################################################
# Assessment tools
################################################################################

# connectivity also for transition matrix:
is_connected = msmest.is_connected
connected_sets = msmest.connected_sets

# DONE : Martin, Ben
def is_transition_matrix(T, tol=1e-15):
    r"""
    True if T is a transition matrix
    
    Parameters
    ----------
    T : numpy.ndarray, shape(d, d) or scipy.sparse matrix
        Matrix to check
    tol : float
        tolerance to check with
    
    Returns
    -------
    Truth value: bool
        True, if T is positive and normed
        False, otherwise
    
    """
    if issparse(T):
        return sparse.assessment.is_transition_matrix(T, tol)
    elif isdense(T):
        return dense.assessment.is_transition_matrix(T, tol)
    else:
        raise _type_not_supported

is_tmatrix=is_transition_matrix
__all__.append('is_tmatrix')


# DONE: Martin, Ben
def is_rate_matrix(K, tol=1e-15):
    r"""True if K is a rate matrix
    
    Parameters
    ----------
    K : ndarray or scipy.sparse matrix
        Rate matrix
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value: bool
        True, if K is a rate matrix
        False, otherwise
    """
    if issparse(K):
        return sparse.assessment.is_rate_matrix(K, tol)
    elif isdense(K):
        return dense.assessment.is_rate_matrix(K, tol)
    else:
        raise _type_not_supported


# DONE: Martin 
def is_ergodic(T, tol=1e-15):
    r"""True if T is connected (irreducible) and aperiodic.
    
    Parameters
    ----------
    T : ndarray or scipy.sparse matrix
        Transition matrix
    tol : float
        tolerance to check with
    
    Returns
    -------
    Truth value : bool
        True, if T is ergodic
        False, otherwise
    """
    if issparse(T) or isdense(T):
        # T has to be sparse, and will be converted in sparse impl
        return sparse.assessment.is_ergodic(T, tol)
    else:
        raise _type_not_supported


# DONE: Martin
def is_reversible(T, mu=None, tol=1e-15):
    r"""True if T is a transition matrix
    
    Parameters
    ----------
    T : ndarray or scipy.sparse matrix
        Transition matrix
    mu : ndarray
        tests with respect to this stationary distribution
    
    Returns
    -------
    Truth value : bool
        True, if T is reversible
        False, otherwise
    """
    if issparse(T):
        return sparse.assessment.is_reversible(T, mu, tol)
    elif isdense(T):
        return dense.assessment.is_reversible(T, mu, tol)
    else:
        raise _type_not_supported


################################################################################
# Eigenvalues and eigenvectors
################################################################################

# DONE: Ben
def stationary_distribution(T):
    r"""Compute stationary distribution of stochastic matrix T. 
    
    The stationary distribution is the left eigenvector corresponding to the 
    non-degenerate eigenvalue :math:`\lambda=1`.
    
    Parameters
    ----------
    T : numpy array, shape(d,d) or scipy.sparse matrix
        Transition matrix (stochastic matrix).
    
    Returns
    -------
    mu : numpy array, shape(d,)      
        Vector of stationary probabilities.
    
    """
    # is this a transition matrix?
    if not is_transition_matrix(T):
        raise ValueError('Input matrix is not a transition matrix. Cannot compute stationary distribution')
    # is the stationary distribution unique?
    if not is_connected(T):
        raise ValueError('Input matrix is not connected and therefore has no unique stationary distribution. '+
                         'Separate disconnected components and handle their stationary distributions separately')
    # we're good to go...
    if issparse(T):
        return sparse.decomposition.stationary_distribution_from_linearsystem(T)
    elif isdense(T):
        return dense.decomposition.stationary_distribution_from_linearsystem(T)
    else: 
        raise _type_not_supported

statdist=stationary_distribution
__all__.append('statdist')


# DONE: Martin
def eigenvalues(T, k=None, ncv=None):
    r"""Find eigenvalues of the transition matrix.
    
    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix
    k : int (optional)
        Compute the first `k` eigenvalues of `T`
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    w : (M,) ndarray
        Eigenvalues of `T`. If `k` is specified, `w` has
        shape (k,)
    
    """
    if issparse(T):
        return sparse.decomposition.eigenvalues(T, k, ncv=ncv)
    elif isdense(T):
        return dense.decomposition.eigenvalues(T, k)
    else:
        raise _type_not_supported

# DONE: Ben
def timescales(T, tau=1, k=None):
    r"""Compute implied time scales of given transition matrix.
    
    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    tau : int (optional)
        The time-lag (in elementary time steps of the microstate
        trajectory) at which the given transition matrix was
        constructed.
    k : int (optional)
        Compute the first `k` implied time scales.
        
    Returns
    -------
    ts : (M,) ndarray
        The implied time scales of the transition matrix.  If `k` is
        not None then the shape of `ts` is (k,).
    
    """
    if issparse(T):
        return sparse.decomposition.timescales(T, tau=tau, k=k)
    elif isdense(T):
        return dense.decomposition.timescales(T, tau=tau, k=k)
    else:
        raise _type_not_supported

# DONE: Ben
def eigenvectors(T, k=None, right=True, ncv=None):
    r"""Compute eigenvectors of given transition matrix.
    
    Eigenvectors are computed using the scipy interface 
    to the corresponding LAPACK/ARPACK routines.    
    
    Parameters
    ----------
    T : numpy.ndarray, shape(d,d) or scipy.sparse matrix
        Transition matrix (stochastic matrix)
    k : int (optional)
        Compute the first k eigenvectors
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than `k`;
        it is recommended that `ncv > 2*k`

    
    Returns
    -------
    eigvec : numpy.ndarray, shape=(d, n)
        The eigenvectors of T ordered with decreasing absolute value of
        the corresponding eigenvalue. If k is None then n=d, if k is
        int then n=k.
    

    Notes
    ------
    The returned eigenvectors :math:`v_i` are normalized such that 

    ..  math::

        \langle v_i, v_j \rangle = \delta_{i,j}

    This is the case for right eigenvectors :math:`r_i` as well as
    for left eigenvectors :math:`l_i`. 

    If you desire orthonormal left and right eigenvectors please use the
    rdl_decomposition method.

    See also
    --------
    rdl_decomposition

    """
    if issparse(T):
        return sparse.decomposition.eigenvectors(T, k=k, right=right, ncv=ncv)
    elif isdense(T):
        return dense.decomposition.eigenvectors(T, k=k, right=right)
    else: 
        raise _type_not_supported

# DONE: Ben
def rdl_decomposition(T, k=None, norm='standard'):
    r"""Compute the decomposition into left and right eigenvectors.
    
    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix    
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible'}, optional
        which normalization convention to use

        ============ ===========================================
        norm       
        ============ ===========================================
        'standard'   L'R = Id, is a probability\
                     distribution, the stationary distribution\
                     of `T`. Right eigenvectors `R`\
                     have a 2-norm of 1
        'reversible' `R` and `L` are related via ``L[:,0]*R``  
        ============ ===========================================        
    
    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, each repeated according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the 
        column ``L[:,i]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[:,i], T)``=``w[i]*L[:,i]``
    R : (M, M) ndarray
        The normalized ("unit length") right eigenvectors, such that the 
        column ``R[:,i]`` is the right eigenvector corresponding to the eigenvalue 
        ``w[i]``, ``dot(T,R[:,i])``=``w[i]*R[:,i]``    

    """    
    if issparse(T):
        return sparse.decomposition.rdl_decomposition(T, k=k, norm=norm)
    elif isdense(T):
        return dense.decomposition.rdl_decomposition(T, k=k, norm=norm)
    else: 
        raise _type_not_supported

# DONE: Ben
def mfpt(T, target):
    r"""Compute vector of mean first passage times to target state.
    
    Parameters
    ----------
    T : ndarray, shape=(n,n) 
        Transition matrix.
    target : int
        Target state for mfpt calculation.
    
    Returns
    -------
    m_t : ndarray, shape=(n,)
         Vector of mean first passage times to target state t.
    
    """
    if issparse(T):
        return sparse.mean_first_passage_time.mfpt(T, target)
    elif isdense(T):
        return dense.mean_first_passage_time.mfpt(T, target)
    else:
        raise _type_not_supported

################################################################################
# Expectations
################################################################################


# DONE: frank
def expectation(T, a, mu=None):
    r"""Equilibrium expectation value of a given observable.

    The expectation value of an observable a is defined as follows
    
    .. math::

        <a> = \sum_i \pi_i a_i
    
    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    a : (M,) ndarray
        Observable vector
    mu : (M,) ndarray (optional)
        The stationary distribution of T.  If given, the stationary
        distribution will not be recalculated (saving lots of time)
    
    Returns
    -------
    val: float
        The expectation value fo the given observable:  <a> = <pi,a> = sum_i pi_i a_i
        <a> = <pi,a> = sum_i pi_i a_i
    """
    pi = stationary_distribution(T)
    return np.dot(pi,a)

# DONE: Ben
def expected_counts(p0, T, n):
    r"""Compute expected transition counts for Markov chain with n steps. 
    
    Expected counts are given by
    
    .. math::
    
        E[C^{(n)}]=\sum_{k=0}^{n-1} \text{diag}(p^{T} T^{k}) T
    
    Parameters
    ----------
    p0 : (M,) ndarray
        Initial (probability) vector
    T : (M, M) ndarray or sparse matrix
        Transition matrix
    n : int
        Number of steps to take
    
    Returns
    --------
    EC : (M, M) ndarray or sparse matrix
        Expected value for transition counts after n steps
    
    """
    if issparse(T):
        return sparse.expectations.expected_counts(p0, T, n)
    elif isdense(T):
        return dense.expectations.expected_counts(p0, T, n)
    else:
        _type_not_supported

# DONE: Ben
def expected_counts_stationary(T, n, mu=None):
    r"""Expected transition counts for Markov chain in equilibrium. 
    
    Since mu is stationary for T we have 
    
    .. math::
    
        E(C^{(N)})=N diag(mu)*T.
    
    Parameters
    ----------
    T : (M, M) ndarray or sparse matrix
        Transition matrix.
    n : int
        Number of steps for chain.
    mu : (M,) ndarray (optional)
        Stationary distribution for T. If mu is not specified it will be
        computed via diagonalization of T.
    
    Returns
    -------
    EC : (M, M) ndarray or sparse matrix
        Expected value for transition counts after N steps.         
    
    """
    if issparse(T):
        return sparse.expectations.expected_counts_stationary(T, n, mu=mu)
    elif isdense(T):
        return dense.expectations.expected_counts_stationary(T, n, mu=mu)
    else:
        _type_not_supported   


################################################################################
# Fingerprints
################################################################################

# DONE: Martin+Frank: Implement in Python directly
def fingerprint_correlation(P, obs1, obs2=None, tau=1):
    r"""Compute dynamical fingerprint crosscorrelation.
    
    The dynamical fingerprint autocorrelation is the timescale
    amplitude spectrum of the autocorrelation of the given observables 
    under the action of the dynamics P
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n,)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n,)
        Vector representing observable 2 on discrete states. 
        If none, obs2=obs1, i.e. the autocorrelation is used
    tau : lag time of the the transition matrix. Used for 
        computing the timescales returned
    
    Returns
    -------
    (timescales, amplitudes)
    timescales : ndarray, shape=(n-1)
        timescales of the relaxation processes of P
    amplitudes : ndarray, shape=(n-1)
        fingerprint amplitdues of the relaxation processes
    
    """
    # handle input
    if (obs2 is None):
        obs2 = obs1
    # rdl_decomposition already handles sparsity of P.
    w, L, R = rdl_decomposition(P)
    # timescales:
    timescales = dense.decomposition.timescales_from_eigenvalues(w, tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(L[i], obs1) * np.dot(L[i], obs2)
    # return
    return timescales, amplitudes


# DONE: Martin+Frank: Implement in Python directly
def fingerprint_relaxation(P, p0, obs, tau=1):
    r"""Compute dynamical fingerprint crosscorrelation.
    
    The dynamical fingerprint autocorrelation is the timescale
    amplitude spectrum of the autocorrelation of the given observables 
    under the action of the dynamics P
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    p0 : ndarray, shape=(n)
        starting distribution
    obs : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. 
        If none, obs2=obs1, i.e. the autocorrelation is used
    tau : lag time of the the transition matrix. Used for 
        computing the timescales returned
    
    Returns
    -------
    (timescales, amplitudes)
    timescales : ndarray, shape=(n-1)
        timescales of the relaxation processes of P
    amplitudes : ndarray, shape=(n-1)
        fingerprint amplitdues of the relaxation processes
    
    """
    # rdl_decomposition already handles sparsity of P.
    w, L, R = rdl_decomposition(P)
    # timescales:
    timescales = dense.decomposition.timescales_from_eigenvalues(w, tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(p0, R[:,i]) * np.dot(L[i], obs)
    # return
    return timescales, amplitudes


# DONE: Frank
def evaluate_fingerprint(timescales, amplitudes, times=[1]):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    
    The time-correlation at time=k is computed by the matrix-vector expression: 
    cor(k) = obs1' diag(pi) P^k obs2
    
    
    Parameters
    ----------
    timescales : ndarray, shape=(n)
        vectors with timescales
    amplitudes : ndarray, shape=(n)
        vector with amplitudes
    times : array-like, shape=(n_t)
        times to evaluate the fingerprint at
    
    Returns
    -------
    
    """
    # check input
    n = len(timescales)
    if (len(amplitudes) != n):
        raise TypeError("length of timescales and amplitudes don't match.")
    n_t = len(times)
    
    # rates
    rates = np.divide(np.ones(n), timescales)
    
    # result
    f = np.zeros(n_t)
    
    for it in range(len(times)):
        t = times[it]
        exponents = -t * rates
        eigenvalues_t = np.exp(exponents)
        f[it] = np.dot(eigenvalues_t, amplitudes)
    
    return f


# DONE: Martin+Frank: Implement in Python directly
def correlation(P, obs1, obs2=None, tau=1, times=[1], pi=None):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    
    The dynamical fingerprint crosscorrelation is the timescale
    amplitude spectrum of the crosscorrelation of the given observables 
    under the action of the dynamics P. The correlation is computed as
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    times : array-like, shape(n_t), type=int or float
        Vector of time points at which the (auto)correlation will be evaluated
    pi : ndarray, shape=(n)
        stationary distribution. If given, it will not be recomputed (much faster!)
    
    Returns
    -------
    
    """
    # observation
    if (obs2 is None):
        obs1 = obs2
    # compute pi if necessary
    if (pi is None):
        pi = stationary_distribution(P)
    # if few and integer time points, compute explicitly
    if (len(times) < 10 and type(sum(times)) == int and isdense(P)):
        f = dense.correlations.time_correlation_direct(P, pi, obs1, obs2, times)
    else:
        timescales,amplitudes = fingerprint_correlation(P, obs1, obs2, tau)
        f = evaluate_fingerprint(timescales, amplitudes, times)
    # return
    return f


# DONE: Martin+Frank: Implement in Python directly
def relaxation(P, p0, obs, tau=1, times=[1], pi=None):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    
    The dynamical fingerprint crosscorrelation is the timescale
    amplitude spectrum of the crosscorrelation of the given observables 
    under the action of the dynamics P. The correlation is computed as
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    times : array-like, shape(n_t), type=int or float
        Vector of time points at which the (auto)correlation will be evaluated
    pi : ndarray, shape=(n)
        stationary distribution. If given, it will not be recomputed (much faster!)
    
    Returns
    -------
    
    """
    # compute pi if necessary
    if (pi is None):
        pi = stationary_distribution(P)
    # if few and integer time points, compute explicitly
    if (len(times) < 10 and type(sum(times)) == int and isdense(P)):
        f = dense.correlations.time_relaxations_direct(P, pi, obs, times)
    else:
        timescales,amplitudes = fingerprint_relaxation(P, pi, obs, tau)
        f = evaluate_fingerprint(timescales, amplitudes, times)
    # return
    return f


################################################################################
# PCCA
################################################################################

# TODO: Implement in Python directly
def pcca(T, n):
    r"""returns an ndarray(m,n) with n membership functions of length m in columns to be in
    correspondence with the column structure of right and left eigenvectors
    
    Parameters
    ----------
    T : transition matrix
    n : number of metastable processes
    
    Returns
    -------
    m : memberships
    
    """
    if issparse(T):
        raise NotImplementedError('not yet impled for sparse.')
    elif isdense(T):
        return dense.pcca.pcca(T, n)
    else:
        _type_not_supported


################################################################################
# Transition path theory
################################################################################

# DONE: Implement in Python directly
def committor(P, A, B, forward=True):
    r"""Compute the committor between sets of microstates.
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    forward : bool
        If True compute the forward committor, else
        compute the backward committor.
    
    Returns
    -------
    x : ndarray, shape=(n, )
        Committor vector.
    
    """
    if issparse(P):
        raise NotImplementedError('not yet impled for sparse.')
    elif isdense(P):
        if forward:
            committor = dense.committor.forward_committor(P, A, B)
        else:
            """ if P is time reversible backward commitor is equal 1 - q+"""
            if is_reversible(P):
                committor = 1.0 - dense.committor.forward_committor(P, A, B)
            else:
                committor = dense.committor.backward_committor(P, A, B)
    else:
        raise _type_not_supported
    
    return committor



# DONE: Martin (sparse implementation missing)
def tpt(T, A, B):
    r""" returns a transition path TPTFlux object.

    Parameters
    ----------
    T : ndarray shape = (n, n)
        transition matrix
    A : ndarray(dtype=int, shape=(n, ))
        cluster centers of set A
    B : cluster centers of set B
        ndarray(dtype=int, shape=(n, ))
    
    Returns
    -------
    tpt : stallone.ITPTFlux
        a transition path TPTFlux object

    Notes
    -----
    invokes stallones (java) markov model factory to create a TPTFlux

    """
    if not is_transition_matrix(T):
        raise ValueError('given matrix T is not a transition matrix')
    
    from _impl import TPTFlux
    return TPTFlux(T, A, B)

################################################################################
# Sensitivities
################################################################################

# TODO: Implement sparse in Python directly
def eigenvalue_sensitivity(T, k):
    r"""Sensitivity matrix of a specified eigenvalue.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int
        Compute sensitivity matrix for k-th eigenvalue

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for k-th eigenvalue.
    
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        return dense.sensitivity.eigenvalue_sensitivity(T, k)
    else:
        raise _type_not_supported

# TODO: Implement sparse in Python directly
def timescale_sensitivity(T, k):
    r"""Sensitivity matrix of a specified time-scale.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    k : int
        Compute sensitivity matrix for the k-th time-scale.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the k-th time-scale.
        
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        return dense.sensitivity.timescale_sensitivity(T, k)
    else:
        raise _type_not_supported

# TODO: Implement sparse in Python directly
def eigenvector_sensitivity(T, k, j, right=True):
    r"""Sensitivity matrix of a selected eigenvector element.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix (stochastic matrix).
    k : int
        Eigenvector index 
    j : int
        Element index 
    right : bool
        If True compute for right eigenvector, otherwise compute for left eigenvector.

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the j-th element of the k-th eigenvector.
    
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        if right is True:
            return dense.sensitivity.eigenvector_sensitivity(T, k, j, True)
        else:
            return dense.sensitivity.eigenvector_sensitivity(T, k, j, False)
    else:
        raise _type_not_supported

# DONE: Implement in Python directly
def stationary_distribution_sensitivity(T, j):
    r"""Sensitivity matrix of a stationary distribution element.
    
    Parameters
    ----------
    T : (M, M) ndarray
       Transition matrix (stochastic matrix).
    j : int
        Index of stationary distribution element
        for which sensitivity matrix is computed.
        

    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for the specified element
        of the stationary distribution.
    
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        return dense.sensitivity.stationary_distribution_sensitivity(T, j)
    else:
        raise _type_not_supported

statdist_sensitivity=stationary_distribution_sensitivity
__all__.append('statdist_sensitivity')

# TODO: Implement sparse in Python directly
def mfpt_sensitivity(T, target, i):
    r"""Sensitivity matrix of the mean first-passage time from specified state.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix 
    target : int or list
        Target state or set for mfpt computation
    i : int
        Compute the sensitivity for state `i`
        
    Returns
    -------
    S : (M, M) ndarray
        Sensitivity matrix for specified state
    
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        return dense.sensitivity.mfpt_sensitivity(T, target, i)
    else:
        raise _type_not_supported

# DONE: Jan (sparse implementation missing)
def committor_sensitivity(T, A, B, i, forward=True):
    r"""Sensitivity matrix of a specified committor entry.
    
    Parameters
    ----------
    
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    i : int
        Compute the sensitivity for committor entry `i`
    forward : bool (optional)
        Compute the forward committor. If forward
        is False compute the backward committor.
    
    Returns
    -------    
    S : (M, M) ndarray
        Sensitivity matrix of the specified committor entry.
    
    """
    if issparse(T):
        raise NotImplementedError('Not implemented.')
    elif isdense(T):
        if forward:
            return dense.sensitivity.forward_committor_sensitivity(T, A, B, i)
        else:
            return dense.sensitivity.backward_committor_sensitivity(T, A, B, i)
    else:
        raise _type_not_supported

# TODO: Implement in Python directly
def expectation_sensitivity(T, a):
    r"""computes the sensitivity of the expectation value of a
    """
    raise NotImplementedError('Not implemented.')
