r"""User-API for the pyemma.msm package

"""

__docformat__ = "restructuredtext en"

from flux import tpt as tpt_factory
from ui import ImpliedTimescales
from ui import MSM
from ui import cktest as chapman_kolmogorov
# from estimation.dense.hidden_markov_model import HiddenMSM

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
         'tpt']

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
    itsobj : :class:`pyemma.msm.ui.ImpliedTimescales` object

    See also
    --------
    pyemma.msm.ui.ImpliedTimescales

    """
    itsobj = ImpliedTimescales(dtrajs, lags=lags, nits=nits, reversible=reversible, connected=connected)
    return itsobj


def msm(dtrajs, lag, reversible=True, sliding=True, compute=True):
    r"""Estimate Markov state model (MSM) from discrete trajectories.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    lag : int
        lagtime for the MSM estimation in multiples of trajectory steps
    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM
    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived quantities using sparse matrix algebra.
        In this case python sparse matrices will be returned by the corresponding functions instead of numpy
        arrays. This behavior is suggested for very large numbers of states (e.g. > 4000) because it is likely
        to be much more efficient.
    connectivity : str, optional, default = 'largest'
        Connectivity mode. Three methods are intended (currently only 'largest' is implemented)
        'largest' : The active set is the largest reversibly connected set. All estimation will be done on this
            subset and all quantities (transition matrix, stationary distribution, etc) are only defined on this
            subset and are correspondingly smaller than the full set of states
        'all' : The active set is the full set of states. Estimation will be conducted on each reversibly connected
            set separately. That means the transition matrix will decompose into disconnected submatrices,
            the stationary vector is only defined within subsets, etc. Currently not implemented.
        'none' : The active set is the full set of states. Estimation will be conducted on the full set of states
            without ensuring connectivity. This only permits nonreversible estimation. Currently not implemented.
    compute : bool, optional, default=True
        If true estimate the MSM when creating the MSM object.
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):
        'fs',  'femtosecond*'
        'ps',  'picosecond*'
        'ns',  'nanosecond*'
        'us',  'microsecond*'
        'ms',  'millisecond*'
        's',   'second*'

    **kwargs: Optional algorithm-specific parameters. See below for special cases
    maxiter = 1000000 : int
        Optional parameter with reversible = True.
        maximum number of iterations before the transition matrix estimation method exits
    maxerr = 1e-8 : float
        Optional parameter with reversible = True.
        convergence tolerance for transition matrix estimation.
        This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
        :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.

    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.compute()
    method.

    See also
    --------
    pyemma.msm.ui.MSM

    """

    msmobj = MSM(dtrajs, lag, reversible=reversible, sliding=sliding, compute=compute)
    return msmobj


def cktest(msmobj, K, nsets=2, sets=None, full_output=False):
    r"""Perform Chapman-Kolmogorov tests for given data.

    Parameters
    ----------
    msmobj : :class:`pyemma.msm.ui.MSM` object
        Markov state model (MSM) object
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
    set_factors : (K, nsets) ndarray, optional
        set_factor[k, i] is the quotient of the MD and the MSM set probabilities

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
    """
    P = msmobj.transition_matrix
    lcc = msmobj.largest_connected_set
    dtrajs = msmobj.discrete_trajectories
    tau = msmobj.lagtime
    return chapman_kolmogorov(P, lcc, dtrajs, tau, K, 
                              nsets=nsets, sets=sets, full_output=full_output)

def tpt(msmobj, A, B):
    r"""Computes the A->B reactive flux using transition path theory (TPT)
    
    Parameters
    ----------
    msmobj : :class:`pyemma.msm.ui.MSM` object
        Markov state model (MSM) object
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
        
    Returns
    -------
    tptobj : :class:`pyemma.msm.flux.ReactiveFlux` object
        A python object containing the reactive A->B flux network
        and several additional quantities, such as stationary probability,
        committors and set definitions
        
    Notes
    -----
    The central object used in transition path theory is
    the forward and backward comittor function.
    
    TPT (originally introduced in [1]) for continuous systems has a
    discrete version outlined in [2]. Here, we use the transition
    matrix formulation described in [3].
    
    See also
    --------
    pyemma.msm.flux.ReactiveFlux
    
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
    T = msmobj.transition_matrix
    mu = msmobj.stationary_distribution
    tptobj = tpt_factory(T, A, B, mu=mu)
    return tptobj   


# def hmsm(dtrajs, nstate, lag=1, conv=0.01, maxiter=None, timeshift=None,
#          TCinit = None, chiInit = None):
#     """
#     Implements a discrete Hidden Markov state model of conformational
#     kinetics.  For details, see [1]_.

#     .. [1] Noe, F. and Wu, H. and Prinz, J.-H. and Plattner, N. (2013)
#     Projected and Hidden Markov Models for calculating kinetics and
#     metastable states of complex molecules.  J. Chem. Phys., 139. p. 184114

#     Parameters
#     ----------
#     dtrajs : int-array or list of int-arrays
#         discrete trajectory or list of discrete trajectories
#     nstate : int
#         number of hidden states
#     lag : int
#         lag time at which the hidden transition matrix will be
#         estimated
#     conv = 0.01 : float
#         convergence criterion. The EM optimization will stop when the
#         likelihood has not increased by more than conv.
#     maxiter : int
#         maximum number of iterations until the EM optimization will be
#         stopped even when no convergence is achieved. By default, will
#         be set to 100 * nstate^2
#     timeshift : int
#         time-shift when using the window method for estimating at lag
#         times > 1. For example, when we have lag = 10 and timeshift =
#         2, the estimation will be conducted using five subtrajectories
#         with the following indexes:
#         [0, 10, 20, ...]
#         [2, 12, 22, ...]
#         [4, 14, 24, ...]
#         [6, 16, 26, ...]
#         [8, 18, 28, ...]
#         Basicly, when timeshift = 1, all data will be used, while for
#         > 1 data will be subsampled. Setting timeshift greater than
#         tau will have no effect, because at least the first
#         subtrajectory will be used.
#     TCinit : ndarray (m,m)
#         initial hidden transition matrix. If set to None, will generate a guess
#         using PCCA+ from a Markov model of the discrete trajectories estimated
#         at the given lag time.
#     chiInit : ndarray (m,n)
#         initial observation probability matrix. If set to None, will generate
#         a guess using PCCA+ from a Markov model of the discrete trajectories
#         estimated at the given lag time.

#     Returns
#     -------
#     hmsm obj : :class:`pyemma.msm.estimation.dense.hidden_markov.model.HiddenMSM`
#        instance.

#     """
#     # initialize
#     return HiddenMSM(dtrajs, nstate, lag=lag, conv=conv, maxiter=maxiter,
#                          timeshift=timeshift,
#                          TCinit=TCinit, chiInit=chiInit)

