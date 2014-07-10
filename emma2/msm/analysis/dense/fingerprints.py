r"""This module provides dense implementation for the computation of
dynamical fingerprints, expectations and correlations

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: M.Scherer <m.scherer AT fu-berlin DOT de>

"""

import numpy as np
from scipy.sparse.sputils import isdense

from decomposition import rdl_decomposition, timescales_from_eigenvalues
from decomposition import stationary_distribution_from_backward_iteration as statdist

def fingerprint(P, obs1, obs2=None, p0=None, k=None, tau=1):
    r"""Dynamical fingerprint for equilibrium or relaxation experiment

    The dynamical fingerprint is given by the implied time-scale
    spectrum together with the corresponding amplitudes.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    k : int (optional)
        Number of time-scales and amplitudes to compute
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales

    Returns
    -------
    timescales : (N,) ndarray
        Time-scales of the transition matrix
    amplitudes : (N,) ndarray
        Amplitudes for the given observable(s)
        
    """
    if obs2 is None: 
        obs2=obs1
    R, D, L=rdl_decomposition(P, k=k)            
    """Extract diagonal"""
    w=np.diagonal(D)  
    """Compute time-scales"""
    timescales = timescales_from_eigenvalues(w, tau)      
    if p0 is None:
        """Use only left eigenvectors"""
        amplitudes=np.dot(L, obs1)*np.dot(L, obs2)
    else:
        """Use initial distribution"""
        amplitudes=np.dot(p0*obs1, R)*np.dot(L, obs2)
    return timescales, amplitudes

def evaluate_fingerprint(timescales, amplitudes, times=[1]):
    r"""Evaluate fingerprint result.

    Parameters
    ----------
    timescales : (M,) ndarray
        Implied time-scales
    amplitudes : (M, ) ndarray
        Amplitudes
    times : list
        List of times in (tau) at which to evaluate fingerprint
        
    Returns
    -------
    res : ndarray
        Array of fingerprint evaluations
        
    """
    if timescales.shape != amplitudes.shape:
        raise ValueError("Shapes of timescales and amplitudes don't match")
    times=np.asarray(times)
    """Compute eigenvalues at all times"""
    eigenvalues_t=np.e**(-times[:,np.newaxis]/timescales[np.newaxis,:])
    """Compute result"""
    res=np.dot(eigenvalues_t, amplitudes)
    return res

def expectation(P, obs):
    r"""Equilibrium expectation of given observable.
    
    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs : (M,) ndarray
        Observable, represented as vector on state space

    Returns
    -------
    x : float
        Expectation value    

    """
    pi=statdist(P)
    return np.dot(pi, obs)

def correlation(P, obs1, obs2=None, k=None, tau=1, times=[1]):
    r"""Time-correlation for equilibrium experiment.
    
    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    obs1 : (M,) ndarray
        Observable, represented as vector on state space
    obs2 : (M,) ndarray (optional)
        Second observable, for cross-correlations
    k : int (optional)
        Number of time-scales and amplitudes to use for computation
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales    
    times : list (optional)
        List of times in (tau) at which to compute correlation

    Returns
    -------
    correlations : ndarray
        Correlation values at given times
        
    """
    timescales, amplitudes=fingerprint(P, obs1, obs2=obs2, k=k, tau=tau)
    res=evaluate_fingerprint(timescales, amplitudes, times)
    return res       

def relaxation(P, p0, obs, k=None, tau=1, times=[1]):
    r"""Relaxation experiment.

    The relaxation experiment describes the time-evolution
    of an expectation value starting in a non-equilibrium
    situation.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    p0 : (M,) ndarray (optional)
        Initial distribution for a relaxation experiment
    obs : (M,) ndarray
        Observable, represented as vector on state space
    k : int (optional)
        Number of time-scales and amplitudes to compute
    tau : int (optional)
        Lag time of given transition matrix, for correct time-scales
    times : list
        List of times in (tau) at which to compute expectation

    Returns
    -------
    res : ndarray
        Array of expectation value at given times
        
    """
    one_vec=np.ones_like(obs) 
    timescales, amplitudes=fingerprint(P, one_vec, obs2=obs, p0=p0, k=k, tau=tau)
    res=evaluate_fingerprint(timescales, amplitudes, times)
    return res                



