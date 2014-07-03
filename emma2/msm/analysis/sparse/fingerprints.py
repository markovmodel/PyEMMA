r"""This module contains sparse implementation of the fingerprint module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np

from decomposition import rdl_decomposition, timescales_from_eigenvalues

def fingerprint_correlation(P, obs1, obs2=None, tau=1, k=None, ncv=None):
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
    k : int (optional)
        Number of amplitudes
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k       
    
    Returns
    -------
    timescales : ndarray, shape=(n-1)
        timescales of the relaxation processes of P
    amplitudes : ndarray, shape=(n-1)
        fingerprint amplitdues of the relaxation processes
    
    """
    # handle input
    if obs2 is None:
        obs2 = obs1
    R, D, L=rdl_decomposition(P, k=k, ncv=ncv)
    w=np.diagonal(D)
    # timescales:
    timescales = timescales_from_eigenvalues(w, tau=tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(L[i], obs1) * np.dot(L[i], obs2)
    # return
    return timescales, amplitudes

def fingerprint_relaxation(P, p0, obs, tau=1, k=None, ncv=None):
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
    R, D, L=rdl_decomposition(P, k=k, ncv=ncv)
    w=np.diagonal(D)
    # timescales:
    timescales = timescales_from_eigenvalues(w, tau)
    n = len(timescales)
    # amplitudes:
    amplitudes = np.zeros(n)
    for i in range(n):
        amplitudes[i] = np.dot(p0, R[:,i]) * np.dot(L[i], obs)
    # return
    return timescales, amplitudes
