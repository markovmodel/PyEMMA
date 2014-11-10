r"""User-API for the pyemma.msm package

"""

__docformat__ = "restructuredtext en"

from flux import tpt as tpt_factory

from ui.timescales import ImpliedTimescales
from ui.msm import MSM
from ui.chapman_kolmogorov import chapman_kolmogorov
from estimation.dense import hidden_markov_model as hmm

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__=['its',
         'msm',
         'cktest',
         'tpt',
         'hmsm']

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

def msm(dtrajs, lag, reversible=True, sliding=True, compute=True):
    r"""Estimate Markov state model (MSM) from discrete trajectories.
    
    Parameters
    ----------
    dtrajs : list
        discrete trajectories
    lag : int
        lagtime for the MSM estimation
    reversible : bool, optional
        If true compute reversible MSM, else non-reversible MSM
    sliding : bool, optional
        If true use the sliding approach to counting, else
        use the lagsampling approach
    compute : bool, optional
        If true estimate the MSM when creating the MSM object

    Returns
    -------
    msmobj : pyemma.msm.ui.msm.MSM object
        A python object containing the MSM and important quantities
        derived from it
        
    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.compute()
    method.       
    
    """
    msmobj=MSM(dtrajs, lag, reversible=reversible, sliding=sliding, compute=compute)
    return msmobj   

def cktest(dtrajs, lag, K, nsets=2, sets=None):
    r"""Perform Chapman-Kolmogorov tests for given data.

    Parameters
    ----------
    dtrajs : list
        discrete trajectories
    lag : int
        lagtime for the MSM estimation
    K : int 
        number of time points for the test
    nsets : int, optional
        number of PCCA sets on which to perform the test
    sets : list, optional
        List of user defined sets for the test

    Returns
    -------
    p_MSM : (K, n_sets) ndarray
        p_MSM[k, l] is the probability of making a transition from
        set l to set l after k*lag steps for the MSM computed at 1*lag
    p_MD : (K, n_sets) ndarray
        p_MD[k, l] is the probability of making a transition from
        set l to set l after k*lag steps as estimated from the given data
    eps_MD : (K, n_sets)
        eps_MD[k, l] is an estimate for the statistical error of p_MD[k, l]   

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
        
    """
    return chapman_kolmogorov(dtrajs, lag, K, nsets=nsets, sets=sets)

def tpt(dtrajs, lag, A, B, reversible=True, sliding=True):
    r""" Computes the A->B reactive flux using transition path theory (TPT)  
    
    Parameters
    ----------
    dtrajs : list
        discrete trajectories
    lag : int
        lagtime for the MSM estimation
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    reversible : bool, optional
        If true compute reversible MSM, else non-reversible MSM
    sliding : bool, optional
        If true use the sliding approach to counting, else
        use the lagsampling approach

    Returns
    -------
    tptobj : pyemma.msm.flux.ReactiveFlux object
        A python object containing the reactive A->B flux network
        and several additional quantities, such as stationary probability,
        committors and set definitions.

    Notes
    -----
    The central object used in transition path theory is
    the forward and backward comittor function. 

    TPT (originally introduced in [1]) for continous systems has a
    discrete version outlined in [2]. Here, we use the transition
    matrix formulation described in [3].

    References
    ----------
    .. [1] W. E and E. Vanden-Eijnden.
        Towards a theory of transition paths. 
        J. Stat. Phys. 123: 503-523 (2006)
    .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
    .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    msmobj=MSM(dtrajs, lag, reversible=reversible, sliding=sliding)
    T=msmobj.transition_matrix
    mu=msmobj.stationary_distribution
    tptobj=tpt_factory(T, A, B, mu=mu)
    return tptobj   

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

