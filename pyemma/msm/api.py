'''
Created on Sep 9, 2014

@author: noe
'''

__all__=['its',
         'hmsm']

import estimation.dense.hidden_markov_model as hmm

from ui.timescales import ImpliedTimescales

def its(dtrajs, lags = None, nits=10, reversible = True, connected = True):
    r"""Calculates the implied timescales for a series of lag times.
        
    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories
    lags : array-like of integers (optional)
        integer lag times at which the implied timescales will be
        calculated
    k : int (optional)
        number of implied timescales to be computed. Will compute less
        if the number of states are smaller
    connected : boolean (optional)
        If true compute the connected set before transition matrix
        estimation at each lag separately
    reversible : boolean (optional)
        Estimate the transition matrix reversibly (True) or
        nonreversibly (False)

    Returns
    -------
    lagtimes : list
        List of lagtimes for which timescales were computed
    timescales : (L, K) ndarray
        The array of implied time-scales. L is the number of lagtimes and
        K is the number of the computed time-scale.
        
    """
    itsobj=ImpliedTimescales(dtrajs, lags=lags, nits=nits, reversible=reversible, connected=connected)
    lagtimes=itsobj.get_lagtimes()
    timescales=itsobj.get_timescales()
    return lagtimes, timescales

# def msm(dtrajs, lag = 1, reversible = True, connected = True):
#     """
#     """
#     #

# def cktest(dtrajs, msm, nsets = 2, sets = None):
#     """
#     """
#     #

# def tpt():
#     """
#     """
#     #

def hmsm(dtrajs, nstate, lag = 1, conv = 0.01, maxiter = None, timeshift = None):
    """
    Implements a discrete Hidden Markov state model of conformational
    kinetics.  For details, see [1].
    
    [1]_ Noe, F. and Wu, H. and Prinz, J.-H. and Plattner, N. (2013)
    Projected and Hidden Markov Models for calculating kinetics and
    metastable states of complex molecules.  J. Chem. Phys., 139
    . p. 184114
    
    Parameters
    ----------
    dtrajs : int-array or list of int-arrays
        discrete trajectory or list of discrete trajectories
    nstate : int
        number of hidden states
    lag : int
        lag time at which the hidden transition matrix will be
        estimated
    conv = 0.01 : float
        convergence criterion. The EM optimization will stop when the
        likelihood has not increased by more than conv.
    maxiter : int
        maximum number of iterations until the EM optimization will be
        stopped even when no convergence is achieved. By default, will
        be set to 100 * nstate^2
    timeshift : int
        time-shift when using the window method for estimating at lag
        times > 1. For example, when we have lag = 10 and timeshift =
        2, the estimation will be conducted using five subtrajectories
        with the following indexes:
        [0, 10, 20, ...]
        [2, 12, 22, ...]
        [4, 14, 24, ...]
        [6, 16, 26, ...]
        [8, 18, 28, ...]
        Basicly, when timeshift = 1, all data will be used, while for
        > 1 data will be subsampled. Setting timeshift greater than
        tau will have no effect, because at least the first
        subtrajectory will be used.
        
    """
    # initialize
    return hmm.HiddenMSM(dtrajs, nstate, lag = lag, conv = conv, maxiter = maxiter, timeshift = timeshift)

