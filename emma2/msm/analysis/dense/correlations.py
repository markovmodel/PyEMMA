'''
Created on 29.11.2013

@author: marscher, noe
'''

import numpy as np

import scipy
from scipy.sparse import issparse
from scipy.sparse.sputils import isdense


def time_correlation_direct(P, pi, obs1, obs2=None, time=1):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    
    The time-correlation at time=k is computed by the matrix-vector expression: 
    cor(k) = obs1' diag(pi) P^k obs2
    
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    pi : ndarray, shape=(n)
        stationary distribution vector. Will be computed if not given
    time : int or array like
        time point(s) at which the (auto)correlation will be evaluated 
    
    Returns
    -------
    
    """
    if not (type( time ) == int):
        raise TypeError("time is not an integer: "+str(time))
    # multiply element-wise obs1 and pi. this is obs1' diag(pi)
    l = np.multiply(obs1, pi)
    # raise transition matrix to power of time
    Pk = np.linalg.matrix_power(P, time)
    # compute product P^k obs2
    r = np.dot(Pk, obs2)
    # return result
    return np.dot(l,r)


def time_correlations_direct(P, pi, obs1, obs2=None, times):
    r"""Compute time-correlation of obs1, or time-cross-correlation with obs2.
    
    The time-correlation at time=k is computed by the matrix-vector expression: 
    cor(k) = obs1' diag(pi) P^k obs2
    
    
    Parameters
    ----------
    P : ndarray, shape=(n, n) or scipy.sparse matrix
        Transition matrix
    obs1 : ndarray, shape=(n)
        Vector representing observable 1 on discrete states
    obs2 : ndarray, shape=(n)
        Vector representing observable 2 on discrete states. If not given,
        the autocorrelation of obs1 will be computed
    pi : ndarray, shape=(n)
        stationary distribution vector. Will be computed if not given
    times : array-like, shape(n_t)
        Vector of time points at which the (auto)correlation will be evaluated 
    
    Returns
    -------
    
    """
    n_t = len(times)
    f = np.zeros((n_t))
    for i in range(n_t):
        f[i] = time_correlation_direct(P, pi, obs1, obs2, times[i])
    return f


