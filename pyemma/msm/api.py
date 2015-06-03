
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

r"""User API for the pyemma.msm package

"""

__docformat__ = "restructuredtext en"

from pyemma.msm.ui.hmsm_estimator import HMSMEstimator as _HMSMEstimator
from pyemma.msm.ui.msm_bayesian_estimator import BayesianMSMEstimator as _BayesianMSMEstimator
from pyemma.msm.ui.hmsm_bayesian_estimator import BayesianHMSMEstimator as _BayesianHMSMEstimator

from flux import tpt as tpt_factory
from pyemma.msm.estimators.msm_estimator import MSMEstimator as _MSMEstimator
from ui import ImpliedTimescales
from ui import MSM
from ui import cktest as chapman_kolmogorov


__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['its',
           'markov_model',
           'estimate_markov_model',
           'bayesian_markov_model',
           'estimate_hidden_markov_model',
           'bayesian_hidden_markov_model',
           'cktest',
           'tpt']


# TODO: need code examples
def its(dtrajs, lags=None, nits=10, reversible=True, connected=True):
    r"""Calculate implied timescales for a series of lag times.

    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories
    lags : array-like of integers (optional)
        integer lag times at which the implied timescales will be
        calculated
    nits : int (optional)
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
    itsobj : :class:`ImpliedTimescales <pyemma.msm.ui.ImpliedTimescales>` object

    See also
    --------
    ImpliedTimescales
        The object returned by this function.
    pyemma.plots.plot_implied_timescales
        Plotting function for the :class:`ImpliedTimescales <pyemma.msm.ui.ImpliedTimescales>` object

    References
    ----------
    .. [1] Swope, W. C. and J. W. Pitera and F. Suits
        Describing protein folding kinetics by molecular dynamics simulations: 1. Theory.
        J. Phys. Chem. B 108: 6571-6581 (2004)

    """
    itsobj = ImpliedTimescales(dtrajs, lags=lags, nits=nits, reversible=reversible, connected=connected)
    return itsobj


# TODO: need code examples
def markov_model(P, dt='1 step'):
    r"""Markov model with a given transition matrix

    Returns a :class:`MSM <pyemma.msm.ui.MSM>` that contains the transition matrix
    and allows to compute a large number of quantities related to Markov models.

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    Returns
    -------
    A :class:`MSM <pyemma.msm.ui.MSM>` object containing a transition matrix and various other MSM-related quantities.

    See also
    --------
    MSM : A MSM object

    """
    return MSM(P, dt=dt)


# TODO: need code examples
def estimate_markov_model(dtrajs, lag, reversible=True, sparse=False, connectivity='largest', dt='1 step', **kwargs):
    r"""Estimates a Markov model from discrete trajectories

    Returns a :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` that contains the estimated transition matrix
    and allows to compute a large number of quantities related to Markov models.

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
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

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

    Returns
    -------
    An :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object containing a transition matrix and various other
    MSM-related quantities.

    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.estimate()
    method.

    See also
    --------
    EstimatedMSM : An MSM object that has been estimated from data

    """
    # transition matrix estimator
    tmestimator = _MSMEstimator(dtrajs, reversible=reversible, sparse=sparse,
                                connectivity=connectivity, dt=dt, **kwargs)
    # estimate and return
    return tmestimator.estimate(lag)


# TODO: need code examples
def estimate_hidden_markov_model(dtrajs, lag, nstates, reversible=True, sparse=False,
                                 connectivity='largest', observe_active=True, dt='1 step', **kwargs):
    r"""Estimates a Markov model from discrete trajectories

    Returns a :class:`EstimatedHMSM <pyemma.msm.ui.EstimatedHMSM>` that contains the estimated hidden transition matrix,
    observation probabilities to the discrete states in dtrajs and allows to compute a large number of quantities
    related to discrete Hidden Markov models.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    lag : int
        lagtime for the MSM estimation in multiples of trajectory steps
    nstates : int
        the number of metastable states in the resulting HMM
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
    observe_active : bool, optional, default=True
        True: Restricts the observation set to the active states of the MSM.
        False: All states are in the observation set.
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

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

    Returns
    -------
    An :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object containing a transition matrix and various other
    HMM-related quantities.

    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.estimate()
    method.

    See also
    --------
    EstimatedHMSM : A discrete HMM object that has been estimated from data

    """
    # estimate MSM
    hmsm_estimator = _HMSMEstimator(dtrajs, reversible=reversible, sparse=sparse, connectivity=connectivity,
                                    observe_active=observe_active, dt=dt, **kwargs)
    return hmsm_estimator.estimate(lag=lag, nstates=nstates)


# TODO: need code examples
def bayesian_markov_model(dtrajs, lag, reversible=True, sparse=False, connectivity='largest',
                          nsample=1000, conf=0.683, dt='1 step', **kwargs):
    r"""Bayesian Markov model estimate using Gibbs sampling of the posterior

    Returns a :class:`SampledMSM <pyemma.msm.ui.SampledMSM>` that contains the estimated transition matrix
    and allows to compute a large number of quantities related to Markov models as well as their statistical
    uncertainties.

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
    nsample : int, optional, default=1000
        number of transition matrix samples to compute and store
    conf : float, optional, default=0.683
        size of confidence intervals
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

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

    Returns
    -------
    An :class:`SampledMSM <pyemma.msm.ui.SampledMSM>` object containing a transition matrix and various other
    MSM-related quantities and statistical uncertainties.

    Notes
    -----
    You can postpone the estimation of the MSM using estimate=False and initiate the estimation procedure by manually
    calling the MSM.estimate() method.
    Likewise, you can postpone the sampling of the MSM using sample=False and initiate the sampling procedure by
    manually calling the MSM.sample() method.

    See also
    --------
    EstimatedMSM : An MSM object that has been estimated from data

    """
    bmsm_estimator = _BayesianMSMEstimator(dtrajs, reversible=reversible, sparse=sparse, connectivity=connectivity,
                                           dt=dt, conf=conf, **kwargs)
    return bmsm_estimator.estimate(lag=lag, nsample=nsample)

# TODO: need code examples
def bayesian_hidden_markov_model(dtrajs, lag, nstates, reversible=True, sparse=False,
                                 connectivity='largest', observe_active=True,
                                 nsample=1000, conf=0.683, dt='1 step', **kwargs):
    r"""Bayesian Hidden Markov model estimate using Gibbs sampling of the posterior

    Returns a :class:`SampledHMSM <pyemma.msm.ui.SampledHMSM>` that contains the estimated hidden Markov model [1]_
    and a Bayesian estimate [2]_ that contains samples around this estimate to estimate uncertainties.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    lag : int
        lagtime for the MSM estimation in multiples of trajectory steps
    nstates : int
        the number of metastable states in the resulting HMM
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
    observe_active : bool, optional, default=True
        True: Restricts the observation set to the active states of the MSM.
        False: All states are in the observation set.
    nsample : int, optional, default=1000
        number of transition matrix samples to compute and store
    conf : float, optional, default=0.683
        size of confidence intervals
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

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

    Returns
    -------
    An :class:`SampledHMSM <pyemma.msm.ui.SampledHMSM>` object containing a transition matrix and various other
    HMM-related quantities and statistical uncertainties.

    Notes
    -----
    You can postpone the estimation of the MSM using estimate=False and initiate the estimation procedure by manually
    calling the MSM.estimate() method.
    Likewise, you can postpone the sampling of the MSM using sample=False and initiate the sampling procedure by
    manually calling the MSM.sample() method.

    See also
    --------
    EstimatedMSM : An MSM object that has been estimated from data

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    .. [2] J. D. Chodera Et Al:
        Bayesian hidden Markov model analysis of single-molecule force spectroscopy:
        Characterizing kinetics under measurement uncertainty
        arXiv:1108.1430 (2011)

    """
    bhmsm_estimator = _BayesianHMSMEstimator(dtrajs, reversible=reversible, sparse=sparse, connectivity=connectivity,
                                             observe_active=observe_active, dt=dt, conf=conf, **kwargs)
    return bhmsm_estimator.estimate(lag=lag, nstates=nstates, nsample=nsample)

# TODO: need code examples
def cktest(msmobj, K, nsets=2, sets=None, full_output=False):
    r"""Chapman-Kolmogorov test for the given MSM

    Parameters
    ----------
    msmobj : :class:`MSM <pyemma.msm.ui.MSM>` or `EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object
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
    dtrajs = msmobj.discrete_trajectories_full
    tau = msmobj.lagtime
    return chapman_kolmogorov(P, lcc, dtrajs, tau, K, nsets=nsets, sets=sets, full_output=full_output)


# TODO: need code examples
def tpt(msmobj, A, B):
    r"""A->B reactive flux from transition path theory (TPT)

    The returned :class:`ReactiveFlux <pyemma.msm.flux.ReactiveFlux>` object can be used to extract various quantities
    of the flux, as well as to compute A -> B transition pathways, their weights, and to coarse-grain the flux onto
    sets of states.

    Parameters
    ----------
    msmobj : :class:`MSM <pyemma.msm.ui.MSM>` or `EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object
        Markov state model (MSM) object
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
        
    Returns
    -------
    tptobj : :class:`ReactiveFlux <pyemma.msm.flux.ReactiveFlux>` object
        A python object containing the reactive A->B flux network
        and several additional quantities, such as stationary probability,
        committors and set definitions.
        
    Notes
    -----
    The central object used in transition path theory is
    the forward and backward committor function.
    
    TPT (originally introduced in [1]_) for continuous systems has a
    discrete version outlined in [2]_. Here, we use the transition
    matrix formulation described in [3]_.
    
    See also
    --------
    ReactiveFlux
        Reactive Flux object
    
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
