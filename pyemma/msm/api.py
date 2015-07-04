
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

from estimators import MaximumLikelihoodHMSM as _ML_HMSM
from estimators import BayesianMSM as _Bayes_MSM
from estimators import BayesianHMSM as _Bayes_HMSM
from estimators import MaximumLikelihoodMSM as _ML_MSM
from estimators import ImpliedTimescales as _ImpliedTimescales

from flux import tpt as tpt_factory
from models import MSM
from util import cktest as chapman_kolmogorov
from pyemma.util.annotators import shortcut
from pyemma.util import types as _types

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['markov_model',
           'timescales_msm',
           'its',
           'estimate_markov_model',
           'bayesian_markov_model',
           'timescales_hmsm',
           'estimate_hidden_markov_model',
           'bayesian_hidden_markov_model',
           'cktest',
           'tpt']


@shortcut('its')
def timescales_msm(dtrajs, lags=None, nits=10, reversible=True, connected=True, errors=None, nsamples=50):
    r""" Calculate implied timescales from Markov state models estimated at a series of lag times.

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
    errors : None or str
        Specifies whether to compute statistical uncertainties (by default
        not), an which algorithm to use if yes.
        Options are 'bayes' for Bayesian sampling of the posterior and
        'bootstrap' for bootstrapping of the discrete trajectories.
        Attention: Computing errors can be *very* slow if the MSM has many
        states. Moreover there are still unsolved theoretical problems, and
        therefore the uncertainty interval and the maximum likelihood
        estimator can be inconsistent. Use this as a rough guess for
        statistical uncertainties.
    nsamples : int
        The number of approximately independent transition matrix samples
        generated for each lag time for uncertainty quantification.
        Only used if errors is not None.

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
    Implied timescales as a lagtime-selection and MSM-validation approach were
    suggested in [1]_. Error estimation is done either using moving block
    bootstrapping [2]_ or a Bayesian analysis using Metropolis-Hastings Monte
    Carlo sampling of the posterior. Nonreversible Bayesian sampling is done
    by independently sampling Dirichtlet distributions of the transition matrix
    rows. A Monte Carlo method for sampling reversible MSMs was introduced
    in [3]_. Here we employ a much more efficient algorithm introduced in [4]_.

    .. [1] Swope, W. C. and J. W. Pitera and F. Suits: Describing protein
        folding kinetics by molecular dynamics simulations: 1. Theory.
        J. Phys. Chem. B 108: 6571-6581 (2004)

    .. [2] Kuensch, H. R.: The jackknife and the bootstrap for general
        stationary observations. Ann. Stat. 17, 1217-1241 (1989)

    .. [3] Noe, F.: Probability Distributions of Molecular Observables computed
        from Markov Models. J. Chem. Phys. 128, 244103 (2008)

    .. [4] Trendelkamp-Schroer et al:
        in preparation (2015)

    Example
    -------
    >>> from pyemma import msm
    >>> dtraj = [0,1,1,2,2,2,1,2,2,2,1,0,0,1,1,1,2,2,1,1,2,1,1,0,0,0,1,1,2,2,1]   # mini-trajectory
    >>> ts = msm.its(dtraj, [1,2,3,4,5])
    >>> print ts.timescales
    [[ 1.50167143  0.20039813]
     [ 3.17036301  1.06407436]
     [ 2.03222416  1.02489382]
     [ 4.63599356  3.42346576]
     [ 5.13829397  2.59477703]]

    """
    # format data
    dtrajs = _types.ensure_dtraj_list(dtrajs)

    if connected:
        connectivity = 'largest'
    else:
        connectivity = 'none'

    # MLE or error estimation?
    if errors is None:
        estimator = _ML_MSM(reversible=reversible, connectivity=connectivity)
    elif errors == 'bayes':
        estimator = _Bayes_MSM(reversible=reversible, connectivity=connectivity)
    else:
        raise NotImplementedError('Error estimation method'+errors+'currently not implemented')

    # go
    itsobj = _ImpliedTimescales(estimator, lags=lags, nits=nits)
    itsobj.estimate(dtrajs)
    return itsobj


def markov_model(P, dt_model='1 step'):
    r""" Markov model with a given transition matrix

    Returns a :class:`MSM <pyemma.msm.ui.MSM>` that contains the transition matrix
    and allows to compute a large number of quantities related to Markov models.

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix

    dt_model : str, optional, default='1 step'
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

    References
    ----------
    Markov chains and theory for analyzing them have been pioneered by A. A. Markov. There are many excellent books
    on the topic, such as [1]_

    .. [1] Norris, J. R.:
        Markov Chains
        Cambridge Series in Statistical and Probabilistic Mathematics, Cambridge University Press (1997)

    Example
    -------
    >>> from pyemma import msm
    >>> import numpy as np
    >>>
    >>> P = np.array([[0.9, 0.1, 0.0], [0.05, 0.94, 0.01], [0.0, 0.02, 0.98]])
    >>> mm = msm.markov_model(P)

    Now we can compute various quantities, e.g. the stationary (equilibrium) distribution:

    >>> print mm.stationary_distribution
    [ 0.25  0.5   0.25]

    The (implied) relaxation timescales

    >>> print mm.timescales
    [ 38.00561796   5.9782565 ]

    The mean first passage time from state 0 to 2

    >>> print mm.mfpt(0, 2)
    160.0

    And many more. See :class:`MSM <pyemma.msm.ui.MSM>` for a full documentation.

    """
    return MSM(P, dt_model=dt_model)


def estimate_markov_model(dtrajs, lag, reversible=True, sparse=False, connectivity='largest',
                          dt_traj='1 step', maxiter=1000000, maxerr=1e-8):
    r""" Estimates a Markov model from discrete trajectories

    Returns a :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` that contains
    the estimated transition matrix and allows to compute a large number of
    quantities related to Markov models.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.

    lag : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.

    connectivity : str, optional, default = 'largest'
        Connectivity mode. Three methods are intended (currently only 'largest'
        is implemented)
        'largest' : The active set is the largest reversibly connected set. All
            estimation will be done on this subset and all quantities
            (transition matrix, stationary distribution, etc) are only defined
            on this subset and are correspondingly smaller than the full set of
            states
        'all' : The active set is the full set of states. Estimation will be
            conducted on each reversibly connected set separately. That means
            the transition matrix will decompose into disconnected submatrices,
            the stationary vector is only defined within subsets, etc.
            Currently not implemented.
        'none' : The active set is the full set of states. Estimation will be
            conducted on the full set of states without ensuring connectivity.
            This only permits nonreversible estimation.
            Currently not implemented.

    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e. there is no physical
        time unit. Specify by a number, whitespace and unit. Permitted units
        are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    maxiter = 1000000 : int
        Optional parameter with reversible = True.
        maximum number of iterations before the transition matrix estimation
        method exits

    maxerr = 1e-8 : float
        Optional parameter with reversible = True.
        convergence tolerance for transition matrix estimation.
        This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative
        stationary probability changes :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})`
        are used in order to track changes in small probabilities. The
        Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to
        maxerr.

    Returns
    -------
    An :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object containing
    a transition matrix and various other MSM-related quantities.

    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.estimate()
    method.

    See also
    --------
    EstimatedMSM : An MSM object that has been estimated from data

    References
    ----------
    The mathematical theory of Markov (state) model estimation was introduced
    in [1]_. Further theoretical developments were made in [2]_. The term
    Markov state model was coined in [3]_. Continuous-time Markov models
    (Master equation models) were suggested in [4]_. Reversible Markov model
    estimation was introduced in [5]_, and further developed in [6]_,[7]_,[9]_.
    It was shown in [8]_ that the quality of Markov state models does in fact
    not depend on memory loss, but rather on where the discretization is
    suitable to approximate the eigenfunctions of the Markov operator (the
    'reaction coordinates'). With a suitable choice of discretization and lag
    time, MSMs can thus become very accurate. [9]_ introduced a number of
    methodological improvements and gives a good overview of the methodological
    basics of Markov state modeling today. [10]_ is a more extensive review
    book of theory, methods and applications.

    .. [1] Schuette, C. , A. Fischer, W. Huisinga and P. Deuflhard:
        A Direct Approach to Conformational Dynamics based on Hybrid Monte
        Carlo. J. Comput. Phys., 151, 146-168 (1999)

    .. [2] Swope, W. C., J. W. Pitera and F. Suits: Describing protein
        folding kinetics by molecular dynamics simulations: 1. Theory
        J. Phys. Chem. B 108, 6571-6581 (2004)

    .. [3] Singhal, N., C. D. Snow, V. S. Pande: Using path sampling to build
        better Markovian state models: Predicting the folding rate and mechanism
        of a tryptophan zipper beta hairpin. J. Chem. Phys. 121, 415 (2004).

    .. [4] Sriraman, S., I. G. Kevrekidis and G. Hummer, G.
        J. Phys. Chem. B 109, 6479-6484 (2005)

    .. [5] Noe, F.: Probability Distributions of Molecular Observables computed
        from Markov Models. J. Chem. Phys. 128, 244103 (2008)

    .. [6] Buchete, N.-V. and Hummer, G.: Coarse master equations for peptide
        folding dynamics. J. Phys. Chem. B 112, 6057--6069 (2008)

    .. [7] Bowman, G. R., K. A. Beauchamp, G. Boxer and V. S. Pande:
        Progress and challenges in the automated construction of Markov state
        models for full protein systems. J. Chem. Phys. 131, 124101 (2009)

    .. [8] Sarich, M., F. Noe and C. Schuette: On the approximation quality
        of Markov state models. SIAM Multiscale Model. Simul. 8, 1154-1177 (2010)

    .. [9] Prinz, J.-H., H. Wu, M. Sarich, B. Keller, M. Senne, M. Held,
        J. D. Chodera, C. Schuette and F. Noe: Markov models of molecular
        kinetics: Generation and Validation J. Chem. Phys. 134, 174105 (2011)

    .. [10] Bowman, G. R., V. S. Pande and F. Noe:
        An Introduction to Markov State Models and Their Application to Long
        Timescale Molecular Simulation. Advances in Experimental Medicine and
        Biology 797, Springer, Heidelberg (2014)

    Example
    -------
    >>> from pyemma import msm
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.estimate_markov_model(dtrajs, 2)

    Which is the active set of states we are working on?

    >>> print mm.active_set
    [0 1 2]

    Show the count matrix


    >>> print mm.count_matrix_active
    [[ 7.  2.  1.]
     [ 2.  0.  4.]
     [ 2.  3.  9.]]

    Show the estimated transition matrix

    >>> print mm.transition_matrix
    [[ 0.69999998  0.16727717  0.13272284]
     [ 0.38787137  0.          0.61212863]
     [ 0.11948368  0.23765916  0.64285715]]

    Is this model reversible (i.e. does it fulfill detailed balance)?

    >>> print mm.is_reversible
    True

    What is the equilibrium distribution of states?

    >>> print mm.stationary_distribution
    [ 0.39337976  0.16965278  0.43696746]

    Relaxation timescales?

    >>> print mm.timescales
    [ 3.41494424  1.29673294]

    Mean first passage time from state 0 to 2:

    >>> print mm.mfpt(0, 2)
    9.92928837718

    """
    # transition matrix estimator
    mlmsm = _ML_MSM(lag=lag, reversible=reversible, sparse=sparse, connectivity=connectivity, dt_traj=dt_traj,
                          maxiter=maxiter, maxerr=maxerr)
    # estimate and return
    return mlmsm.estimate(dtrajs)


def timescales_hmsm(dtrajs, nstates, lags=None, nits=10, reversible=True, connected=True, errors=None, nsamples=100):
    r""" Calculate implied timescales from Hidden Markov state models estimated at a series of lag times.

    Warning: this can be slow!

    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories

    nstates : int
        number of hidden states

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

    errors : None or str
        Specifies whether to compute statistical uncertainties (by default not), an which algorithm to use if yes.
        The only option is currently 'bayes'. This algorithm is much faster than MSM-based error calculation because
        the involved matrices are much smaller.

    nsamples : int
        The number of approximately independent HMSM samples generated for each lag time for uncertainty
        quantification. Only used if errors is not None.

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
    Implied timescales as a lagtime-selection and MSM-validation approach were suggested in [1]_. Hidden Markov
    state model estimation is done here as described in [2]_. For uncertainty quantification we employ the
    Bayesian sampling algorithm described in [3]_.

    .. [1] Swope, W. C. and J. W. Pitera and F. Suits:
        Describing protein folding kinetics by molecular dynamics simulations: 1. Theory.
        J. Phys. Chem. B 108: 6571-6581 (2004)

    .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)

    .. [3] J. D. Chodera Et Al:
        Bayesian hidden Markov model analysis of single-molecule force spectroscopy:
        Characterizing kinetics under measurement uncertainty
        arXiv:1108.1430 (2011)

    Example
    -------
    >>> from pyemma import msm
    >>> dtraj = [0,1,1,2,2,2,1,2,2,2,1,0,0,1,1,1,2,2,1,1,2,1,1,0,0,0,1,1,2,2,1]   # mini-trajectory
    >>> ts = msm.its(dtraj, [1,2,3,4,5])
    >>> print ts.timescales
    [[ 1.50167143  0.20039813]
     [ 3.17036301  1.06407436]
     [ 2.03222416  1.02489382]
     [ 4.63599356  3.42346576]
     [ 5.13829397  2.59477703]]

    """
    # format data
    dtrajs = _types.ensure_dtraj_list(dtrajs)

    if connected:
        connectivity = 'largest'
    else:
        connectivity = 'none'

    # MLE or error estimation?
    if errors is None:
        estimator = _ML_HMSM(nstates=nstates, reversible=reversible, connectivity=connectivity)
    elif errors == 'bayes':
        estimator = _Bayes_HMSM(nstates=nstates, reversible=reversible, connectivity=connectivity)
    else:
        raise NotImplementedError('Error estimation method'+errors+'currently not implemented')

    # go
    itsobj = _ImpliedTimescales(estimator, lags=lags, nits=nits)
    itsobj.estimate(dtrajs)
    return itsobj

def estimate_hidden_markov_model(dtrajs, nstates, lag, reversible=True, connectivity='largest', observe_active=True,
                                 dt_traj='1 step', accuracy=1e-3, maxit=1000):
    r""" Estimates a Hidden Markov state model from discrete trajectories

    Returns a :class:`EstimatedHMSM <pyemma.msm.ui.EstimatedHMSM>` that contains a transition matrix between a few
    (hidden) metastable states. Each metastable state has a probability distribution of visiting the discrete
    'microstates' contained in the input trajectories. The resulting object is a hidden Markov model that
    allows to compute a large number of quantities.

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

    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e. there is no physical
        time unit. Specify by a number, whitespace and unit. Permitted units
        are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    accuracy : float
        convergence threshold for EM iteration. When two the likelihood does
        not increase by more than accuracy, the iteration is stopped
        successfully.

    maxit : int
        stopping criterion for EM iteration. When so many iterations are
        performed without reaching the requested accuracy, the iteration is
        stopped without convergence (a warning is given)

    Returns
    -------
    An :class:`EstimatedMSM <pyemma.msm.ui.EstimatedMSM>` object containing
    a transition matrix and various other HMM-related quantities.

    Notes
    -----
    You can postpone the estimation of the MSM using compute=False and
    initiate the estimation procedure by manually calling the MSM.estimate()
    method.

    See also
    --------
    EstimatedHMSM : A discrete HMM object that has been estimated from data

    References
    ----------
    [1]_ is an excellent review of estimation algorithms for discrete Hidden
    Markov Models. This function estimates a discrete HMM on the discrete
    input states using the Baum-Welch algorithm [2]_. We use a
    maximum-likelihood Markov state model to initialize the HMM estimation as
    described in [3]_.

    .. [1] L. R. Rabiner: A Tutorial on Hidden Markov Models and Selected
        Applications in Speech Recognition. Proc. IEEE 77, 257-286 (1989)

    .. [2] L. Baum, T. Petrie, G. Soules and N. Weiss N: A maximization
        technique occurring in the statistical analysis of probabilistic
        functions of Markov chains. Ann. Math. Statist. 41, 164-171 (1970)

    .. [3] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
        Markov models for calculating kinetics and  metastable states of
        complex molecules. J. Chem. Phys. 139, 184114 (2013)

    Example
    -------
    >>> from pyemma import msm
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.estimate_hidden_markov_model(dtrajs, 2, 2)

    We have estimated a 2x2 hidden transition matrix between the metastable
    states:

    >>> print mm.transition_matrix
    [[ 0.75703873  0.24296127]
     [ 0.20628204  0.79371796]]

    With the equilibrium distribution:

    >>> print mm.stationary_distribution
    [ 0.45917665  0.54082335]

    The observed states are the three discrete clusters that we have in our
    discrete trajectory:

    >>> print mm.observable_set
    [0 1 2]

    The metastable distributions (mm.metastable_distributions), or equivalently
    the observation probabilities are the probability to be in a given cluster
    ('microstate') if we are in one of the hidden metastable states.
    So it's a 2 x 3 matrix:

    >>> print mm.observation_probabilities
    [[ 0.9620883   0.0379117   0.        ]
     [ 0.          0.28014352  0.71985648]]

    The first metastable state ist mostly in cluster 0, and a little bit in the
    transition state cluster 1. The second metastable state is less well
    defined, but mostly in cluster 2 and less prominently in the transition
    state cluster 1.

    We can print the lifetimes of the metastable states:

    >>> print mm.lifetimes
    [ 7.18543435  8.65699332]

    And the timescale of the hidden transition matrix - now we only have one
    relaxation timescale:

    >>> print mm.timescales
    [ 3.35310468]

    The mean first passage times can also be computed between metastable states:

    >>> print mm.mfpt(0, 1)
    8.23176470249

    """
    # initialize HMSM estimator
    hmsm_estimator = _ML_HMSM(lag=lag, nstates=nstates, reversible=reversible, connectivity=connectivity,
                              observe_active=observe_active, dt_traj=dt_traj, accuracy=accuracy, maxit=maxit)
    # run estimation
    return hmsm_estimator.estimate(dtrajs)


# TODO: need code examples
def bayesian_markov_model(dtrajs, lag, reversible=True, sparse=False, connectivity='largest',
                          nsamples=100, conf=0.95, dt_traj='1 step'):
    r""" Bayesian Markov model estimate using Gibbs sampling of the posterior

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

    nsample : int, optional, default=100
        number of transition matrix samples to compute and store

    conf : float, optional, default=0.95
        size of confidence intervals

    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as
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
    # TODO: store_data=True
    bmsm_estimator = _Bayes_MSM(lag=lag, reversible=reversible, sparse=sparse, connectivity=connectivity,
                                dt_traj=dt_traj, nsamples=nsamples, conf=conf)
    return bmsm_estimator.estimate(dtrajs)


def bayesian_hidden_markov_model(dtrajs, nstates, lag, nsamples=100, reversible=True, connectivity='largest',
                                 observe_active=True, conf=0.95, dt_traj='1 step'):
    r""" Bayesian Hidden Markov model estimate using Gibbs sampling of the posterior

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

    nsamples : int, optional, default=100
        number of transition matrix samples to compute and store

    conf : float, optional, default=0.95
        size of confidence intervals

    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as
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

    Example
    -------

    >>> from pyemma import msm
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.bayesian_hidden_markov_model(dtrajs, 2, 2)

    We compute the stationary distribution (here given by the maximum likelihood estimate), and the 1-sigma
    uncertainty interval. You can see that the uncertainties are quite large (we have seen only very few transitions
    between the metastable states:

    >>> pi = mm.stationary_distribution
    >>> piL,piR = mm.stationary_distribution_conf
    >>> for i in range(2): print pi[i],' -',piL[i],'+',piR[i]
    0.459176653019  - 0.268314552886 + 0.715326151685
    0.540823346981  - 0.284761476984 + 0.731730375713

    Let's look at the lifetimes of metastable states. Now we have really huge uncertainties. In states where
    one state is more probable than the other, the mean first passage time from the more probable to the less
    probable state is much higher than the reverse:

    >>> l = mm.lifetimes
    >>> lL, lR = mm.lifetimes_conf
    >>> for i in range(2): print l[i],' -',lL[i],'+',lR[i]
    7.18543434854  - 6.03617757784 + 80.1298222741
    8.65699332061  - 5.35089540896 + 30.1719505772

    In contrast the relaxation timescale is less uncertain. This is because for a two-state system the relaxation
    timescale is dominated by the faster passage, which is less uncertain than the slower passage time:

    >>> ts = mm.timescales
    >>> tsL,tsR = mm.timescales_conf
    >>> print ts[0],' -',tsL[0],'+',tsR[0]
    3.35310468086  - 2.24574587978 + 8.34383177258


    """

    bhmsm_estimator = _Bayes_HMSM(lag=lag, nstates=nstates, nsamples=nsamples, reversible=reversible,
                                  connectivity=connectivity, observe_active=observe_active, dt_traj=dt_traj, conf=conf)
    return bhmsm_estimator.estimate(dtrajs)


# TODO: need code examples
def cktest(msmobj, K, nsets=2, sets=None, full_output=False):
    r""" Chapman-Kolmogorov test for the given MSM

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
    This test was suggested in [1]_ and described in detail in [2]_.
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
    .. [2] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
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
    r""" A->B reactive flux from transition path theory (TPT)

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
    Transition path theory was introduced for space-continuous dynamical processes, such as Langevin dynamics, in [1]_,
    [2]_ introduces discrete transition path theory for Markov jump processes (Master equation models, rate matrices)
    and pathway decomposition algorithms. [3]_ introduces transition path theory for Markov state models (MSMs)
    and some analysis algorithms. In this function, the equations described in [3]_ are applied.

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
