'''
Created on Jun 4, 2014

@author: marscher
'''

import numpy as np
from scipy.sparse.sputils import isdense

from decomposition import rdl_decomposition, timescales_from_eigenvalues
from decomposition import stationary_distribution_from_backward_iteration as stationary_distribution
from correlations import time_correlations_direct, time_relaxations_direct

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
    if obs2 is None:
        obs2 = obs1
    R, D, L=rdl_decomposition(P)
    w=np.diagonal(D)
    L=np.transpose(L) 
    # timescales:
    timescales = timescales_from_eigenvalues(w, tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(L[i], obs1) * np.dot(L[i], obs2)
    # return
    return timescales, amplitudes

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
    # w, L, R = rdl_decomposition(P)
    R, D, L=rdl_decomposition(P)
    w=np.diagonal(D)
    L=np.transpose(L) # TODO: double transposed here?
    # timescales:
    timescales = timescales_from_eigenvalues(w, tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(p0, R[:,i]) * np.dot(L[i], obs)
    # return
    return timescales, amplitudes
    
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
    if len(amplitudes) != n:
        raise TypeError("length of timescales and amplitudes don't match.")
    n_t = len(times)
    
    # rates
    rates = np.divide(np.ones(n), timescales)
    
    # result
    f = np.zeros(n_t)
    
    for it in xrange(len(times)):
        t = times[it]
        exponents = -t * rates
        eigenvalues_t = np.exp(exponents)
        f[it] = np.dot(eigenvalues_t, amplitudes)
    
    return f

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
    if obs2 is None:
        obs1 = obs2
    # compute pi if necessary
    if pi is None:
        pi = stationary_distribution(P)
    # if few and integer time points, compute explicitly
    # TODO: investigate, if this hard cutoff is really necessary.
    if (len(times) < 10 and type(sum(times)) == int and isdense(P)):
        f = time_correlations_direct(P, pi, obs1, obs2, times)
    else:
        timescales,amplitudes = fingerprint_correlation(P, obs1, obs2, tau)
        f = evaluate_fingerprint(timescales, amplitudes, times)
    # return
    return f

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
    # TODO: parameter p0 unused! shall this be def r(P, obs1, obs2, ...)?
    # compute pi if necessary
    if pi is None:
        pi = stationary_distribution(P)
    # if few and integer time points, compute explicitly
    # TODO: reconsider(measure) if improvements in correlations module allow more direct calculations, btw. this is an arbitrary cutoff...
    if len(times) < 10 and type(sum(times)) == int and isdense(P):
        f = time_relaxations_direct(P, pi, obs, times)
    else:
        timescales, amplitudes = fingerprint_relaxation(P, pi, obs, tau)
        f = evaluate_fingerprint(timescales, amplitudes, times)
    # return
    return f
