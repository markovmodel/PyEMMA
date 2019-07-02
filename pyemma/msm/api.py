
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""User API for the pyemma.msm package

"""

from .estimators import MaximumLikelihoodHMSM as _ML_HMSM
from .estimators import BayesianMSM as _Bayes_MSM
from .estimators import BayesianHMSM as _Bayes_HMSM
from .estimators import MaximumLikelihoodMSM as _ML_MSM
from .estimators import AugmentedMarkovModel as _ML_AMM
from .estimators import OOMReweightedMSM as _OOM_MSM
from .estimators import ImpliedTimescales as _ImpliedTimescales

from .models import MSM
from pyemma.util.annotators import shortcut
from pyemma.util import types as _types

import numpy as _np

__docformat__ = "restructuredtext en"
__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['markov_model',
           'timescales_msm',
           'its',
           'estimate_markov_model',
           'bayesian_markov_model',
           'timescales_hmsm',
           'estimate_hidden_markov_model',
           'estimate_augmented_markov_model',
           'bayesian_hidden_markov_model',
           'tpt']

# =============================================================================
# MARKOV STATE MODELS - flat Markov chains on discrete observation space
# =============================================================================


@shortcut('its')
def timescales_msm(dtrajs, lags=None, nits=None, reversible=True, connected=True, weights='empirical',
                   errors=None, nsamples=50, n_jobs=None, show_progress=True, mincount_connectivity='1/n',
                   only_timescales=False, core_set=None, milestoning_method='last_core'):
    # format data
    r""" Implied timescales from Markov state models estimated at a series of lag times.

    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories

    lags : int, array-like with integers or None, optional
        integer lag times at which the implied timescales will be calculated. If set to None (default)
        as list of lag times will be automatically generated. For a single int, generate a set of lag times starting
        from 1 to lags, using a multiplier of 1.5 between successive lags.

    nits : int, optional
        number of implied timescales to be computed. Will compute less
        if the number of states are smaller. If None, the number of timescales
        will be automatically determined.

    reversible : boolean, optional
        Estimate transition matrix reversibly (True) or nonreversibly (False)

    connected : boolean, optional
        If true compute the connected set before transition matrix estimation
        at each lag separately

    weights : str, optional
        can be used to re-weight non-equilibrium data to equilibrium.
        Must be one of the following:

        * 'empirical': Each trajectory frame counts as one. (default)

        * 'oom': Each transition is re-weighted using OOM theory, see [5]_.

    errors : None | 'bayes', optional
        Specifies whether to compute statistical uncertainties (by default
        not), an which algorithm to use if yes. Currently the only option is:

        * 'bayes' for Bayesian sampling of the posterior

        Attention:
        * The Bayes mode will use an estimate for the effective count matrix
          that may produce somewhat different estimates than the
          'sliding window' estimate used with ``errors=None`` by default.
        * Computing errors can be// slow if the MSM has many states.
        * There are still unsolved theoretical problems in the computation
          of effective count matrices, and therefore the uncertainty interval
          and the maximum likelihood estimator can be inconsistent. Use this
          as a rough guess for statistical uncertainties.

    nsamples : int, optional
        The number of approximately independent transition matrix samples
        generated for each lag time for uncertainty quantification.
        Only used if errors is not None.

    n_jobs : int, optional
        how many subprocesses to start to estimate the models for each lag time.

    show_progress : bool, default=True
        whether to show progress of estimation.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    only_timescales: bool, default=False
        If you are only interested in the timescales and its samples,
        you can consider turning this on in order to save memory. This can be
        useful to avoid blowing up memory with BayesianMSM and lots of samples.

    core_set : None (default) or array like, dtype=int
        Definition of core set for milestoning MSMs.
        If set to None, replaces state -1 (if found in discrete trajectories) and
        performs milestone counting. No effect for Voronoi-discretized trajectories (default).
        If a list or np.ndarray is supplied, discrete trajectories will be assigned
        accordingly.

    milestoning_method : str
        Method to use for counting transitions in trajectories with unassigned frames.
        Currently available:
        |  'last_core',   assigns unassigned frames to last visited core

    Returns
    -------
    itsobj : :class:`ImpliedTimescales <pyemma.msm.estimators.implied_timescales.ImpliedTimescales>` object

    Example
    -------
    >>> from pyemma import msm
    >>> dtraj = [0,1,1,2,2,2,1,2,2,2,1,0,0,1,1,1,2,2,1,1,2,1,1,0,0,0,1,1,2,2,1]   # mini-trajectory
    >>> ts = msm.its(dtraj, [1,2,3,4,5], show_progress=False)
    >>> print(ts.timescales)  # doctest: +ELLIPSIS
    [[ 1.5...  0.2...]
     [ 3.1...  1.0...]
     [ 2.03...  1.02...]
     [ 4.63...  3.42...]
     [ 5.13...  2.59...]]

    See also
    --------
    ImpliedTimescales
        The object returned by this function.
    pyemma.plots.plot_implied_timescales
        Implied timescales plotting function. Just call it with the
        :class:`ImpliedTimescales <pyemma.msm.estimators.ImpliedTimescales>`
        object produced by this function as an argument.


    .. autoclass:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
            :attributes:

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
    .. [4] Trendelkamp-Schroer, B, H. Wu, F. Paul and F. Noe:
        Estimation and uncertainty of reversible Markov models.
        http://arxiv.org/abs/1507.05990
    .. [5] Nueske, F., Wu, H., Prinz, J.-H., Wehmeyer, C., Clementi, C. and Noe, F.:
        Markov State Models from short non-Equilibrium Simulations - Analysis and
         Correction of Estimation Bias J. Chem. Phys. (submitted) (2017)

    """
    # Catch invalid inputs for weights:
    if isinstance(weights, str):
        if weights not in ['empirical', 'oom']:
            raise ValueError("Weights must be either \'empirical\' or \'oom\'")
    else:
        raise ValueError("Weights must be either \'empirical\' or \'oom\'")
    # Set errors to None if weights==oom:
    if weights == 'oom' and (errors is not None):
        errors = None

    # format data
    dtrajs = _types.ensure_dtraj_list(dtrajs)

    if connected:
        connectivity = 'largest'
    else:
        connectivity = 'none'

    # Choose estimator:
    if errors is None:
        if weights == 'empirical':
            estimator = _ML_MSM(reversible=reversible, connectivity=connectivity,
                                core_set=core_set, milestoning_method=milestoning_method)
        else:
            if core_set is not None or any(-1 in d for d in dtrajs):
                raise NotImplementedError('OOM models currently not implemented with '
                                          'milestoning.')
            estimator = _OOM_MSM(reversible=reversible, connectivity=connectivity)
    elif errors == 'bayes':
        if core_set is not None or any(-1 in d for d in dtrajs):
            raise NotImplementedError('Convenience function timescales_msm does not support '
                                      'Bayesian error estimates for core set MSMs.')
        estimator = _Bayes_MSM(reversible=reversible, connectivity=connectivity,
                               nsamples=nsamples, show_progress=show_progress)
    else:
        raise NotImplementedError('Error estimation method {errors} currently not implemented'.format(errors=errors))

    if hasattr(estimator, 'mincount_connectivity'):
        estimator.mincount_connectivity = mincount_connectivity
    # go
    itsobj = _ImpliedTimescales(estimator, lags=lags, nits=nits, n_jobs=n_jobs,
                                show_progress=show_progress, only_timescales=only_timescales)
    itsobj.estimate(dtrajs)
    return itsobj


def markov_model(P, dt_model='1 step'):
    r""" Markov model with a given transition matrix

    Returns a :class:`MSM <pyemma.msm.models.msm.MSM>` that contains the transition matrix
    and allows to compute a large number of quantities related to Markov models.

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix

    dt_model : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    Returns
    -------
    msm : A :class:`MSM <pyemma.msm.models.msm.MSM>` object containing a transition
        matrix and various other MSM-related quantities.

    Example
    -------
    >>> from pyemma import msm
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)
    >>>
    >>> P = np.array([[0.9, 0.1, 0.0], [0.05, 0.94, 0.01], [0.0, 0.02, 0.98]])
    >>> mm = msm.markov_model(P)

    Now we can compute various quantities, e.g. the stationary (equilibrium) distribution:

    >>> print(mm.stationary_distribution)
    [ 0.25  0.5   0.25]

    The (implied) relaxation timescales

    >>> print(mm.timescales())
    [ 38.006   5.978]

    The mean first passage time from state 0 to 2

    >>> print(mm.mfpt(0, 2))
    160.0

    And many more. See :class:`MSM <pyemma.msm.models.MSM>` for a full documentation.


    .. autoclass:: pyemma.msm.models.msm.MSM
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.models.msm.MSM
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.models.msm.MSM
            :attributes:

    References
    ----------
    Markov chains and theory for analyzing them have been pioneered by A. A.
    Markov. There are many excellent books on the topic, such as [1]_

    .. [1] Norris, J. R.: Markov Chains. Cambridge Series in Statistical and
        Probabilistic Mathematics, Cambridge University Press (1997)


    """
    return MSM(P, dt_model=dt_model)


def estimate_markov_model(dtrajs, lag, reversible=True, statdist=None,
                          count_mode='sliding', weights='empirical',
                          sparse=False, connectivity='largest',
                          dt_traj='1 step', maxiter=1000000, maxerr=1e-8,
                          score_method='VAMP2', score_k=10, mincount_connectivity='1/n',
                          core_set=None, milestoning_method='last_core'
    ):
    r""" Estimates a Markov model from discrete trajectories

    Returns a :class:`MaximumLikelihoodMSM` that
    contains the estimated transition matrix and allows to compute a
    large number of quantities related to Markov models.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    lag : int
        lag time at which transitions are counted and the transition matrix is
        estimated.
    reversible : bool, optional
        If true compute reversible MSM, else non-reversible MSM
    statdist : (M,) ndarray, optional
        Stationary vector on the full state-space. Transition matrix
        will be estimated such that statdist is its equilibrium
        distribution.
    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

              .. math::

                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MSM.
        * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
          at time indexes

              .. math::

                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)
    weights : str, optional
        can be used to re-weight non-equilibrium data to equilibrium.
        Must be one of the following:

        * 'empirical': Each trajectory frame counts as one. (default)

        * 'oom': Each transition is re-weighted using OOM theory, see [11]_.

    sparse : bool, optional
        If true compute count matrix, transition matrix and all
        derived quantities using sparse matrix algebra.  In this case
        python sparse matrices will be returned by the corresponding
        functions instead of numpy arrays. This behavior is suggested
        for very large numbers of states (e.g. > 4000) because it is
        likely to be much more efficient.
    connectivity : str, optional
        Connectivity mode. Three methods are intended (currently only
        'largest' is implemented)

        * 'largest' : The active set is the largest reversibly
          connected set. All estimation will be done on this subset
          and all quantities (transition matrix, stationary
          distribution, etc) are only defined on this subset and are
          correspondingly smaller than the full set of states

        * 'all' : The active set is the full set of states. Estimation
          will be conducted on each reversibly connected set
          separately. That means the transition matrix will decompose
          into disconnected submatrices, the stationary vector is only
          defined within subsets, etc. Currently not implemented.

        * 'none' : The active set is the full set of
          states. Estimation will be conducted on the full set of
          states without ensuring connectivity. This only permits
          nonreversible estimation. Currently not implemented.
    dt_traj : str, optional
        Description of the physical time corresponding to the lag. May
        be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e. there is no
        physical time unit.  Specify by a number, whitespace and
        unit. Permitted units are (* is an arbitrary string):

        *  'fs',  'femtosecond*'
        *  'ps',  'picosecond*'
        *  'ns',  'nanosecond*'
        *  'us',  'microsecond*'
        *  'ms',  'millisecond*'
        *  's',   'second*'

    maxiter : int, optional
        Optional parameter with reversible = True.  maximum number of
        iterations before the transition matrix estimation method
        exits
    maxerr : float, optional
        Optional parameter with reversible = True.  convergence
        tolerance for transition matrix estimation.  This specifies
        the maximum change of the Euclidean norm of relative
        stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The
        relative stationary probability changes :math:`e_i =
        (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in
        order to track changes in small probabilities. The Euclidean
        norm of the change vector, :math:`|e_i|_2`, is compared to
        maxerr.

    score_method : str, optional, default='VAMP2'
        Score to be used with MSM score function. Available scores are
        based on the variational approach for Markov processes [13]_ [14]_:

        *  'VAMP1'  Sum of singular values of the symmetrized transition matrix [14]_ .
                    If the MSM is reversible, this is equal to the sum of transition
                    matrix eigenvalues, also called Rayleigh quotient [13]_ [15]_ .
        *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix [14]_ .
                    If the MSM is reversible, this is equal to the kinetic variance [16]_ .

    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    core_set : None (default) or array like, dtype=int
        Definition of core set for milestoning MSMs.
        If set to None, replaces state -1 (if found in discrete trajectories) and
        performs milestone counting. No effect for Voronoi-discretized trajectories (default).
        If a list or np.ndarray is supplied, discrete trajectories will be assigned
        accordingly.

    milestoning_method : str
        Method to use for counting transitions in trajectories with unassigned frames.
        Currently available:
        |  'last_core',   assigns unassigned frames to last visited core

    Returns
    -------
    msm : :class:`MaximumLikelihoodMSM <pyemma.msm.MaximumLikelihoodMSM>`
        Estimator object containing the MSM and estimation information.

    See also
    --------
    MaximumLikelihoodMSM
        An MSM object that has been estimated from data


    .. autoclass:: pyemma.msm.estimators.maximum_likelihood_msm.MaximumLikelihoodMSM
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_msm.MaximumLikelihoodMSM
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_msm.MaximumLikelihoodMSM
            :attributes:


    References
    ----------
    The mathematical theory of Markov (state) model estimation was introduced
    in [1]_ . Further theoretical developments were made in [2]_ . The term
    Markov state model was coined in [3]_ . Continuous-time Markov models
    (Master equation models) were suggested in [4]_. Reversible Markov model
    estimation was introduced in [5]_ , and further developed in [6]_ [7]_ [9]_ .
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

    .. [11] Nueske, F., Wu, H., Prinz, J.-H., Wehmeyer, C., Clementi, C. and Noe, F.:
        Markov State Models from short non-Equilibrium Simulations - Analysis and
         Correction of Estimation Bias J. Chem. Phys. (submitted) (2017)

    .. [12] H. Wu and F. Noe: Variational approach for learning Markov processes
        from time series data (in preparation)

    .. [13] Noe, F. and F. Nueske: A variational approach to modeling slow processes
        in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655 (2013).

    .. [14] Wu, H and F. Noe: Variational approach for learning Markov processes
        from time series data (in preparation)

    .. [15] McGibbon, R and V. S. Pande: Variational cross-validation of slow
        dynamical modes in molecular kinetics, J. Chem. Phys. 142, 124105 (2015)

    .. [16] Noe, F. and C. Clementi: Kinetic distance and kinetic maps from molecular
        dynamics simulation. J. Chem. Theory Comput. 11, 5002-5011 (2015)


    Example
    -------
    >>> from pyemma import msm
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.estimate_markov_model(dtrajs, 2)

    Which is the active set of states we are working on?

    >>> print(mm.active_set)
    [0 1 2]

    Show the count matrix


    >>> print(mm.count_matrix_active)
    [[ 7.  2.  1.]
     [ 2.  0.  4.]
     [ 2.  3.  9.]]

    Show the estimated transition matrix

    >>> print(mm.transition_matrix)
    [[ 0.7    0.167  0.133]
     [ 0.388  0.     0.612]
     [ 0.119  0.238  0.643]]

    Is this model reversible (i.e. does it fulfill detailed balance)?

    >>> print(mm.is_reversible)
    True

    What is the equilibrium distribution of states?

    >>> print(mm.stationary_distribution)
    [ 0.393  0.17   0.437]

    Relaxation timescales?

    >>> print(mm.timescales())
    [ 3.415  1.297]

    Mean first passage time from state 0 to 2:

    >>> print(mm.mfpt(0, 2))  # doctest: +ELLIPSIS
    9.929...

    """
    # Catch invalid inputs for weights:
    if isinstance(weights, str):
        if weights not in ['empirical', 'oom']:
            raise ValueError("Weights must be either \'empirical\' or \'oom\'")
    else:
        raise ValueError("Weights must be either \'empirical\' or \'oom\'")

    # transition matrix estimator
    if weights == 'empirical':
        mlmsm = _ML_MSM(lag=lag, reversible=reversible, statdist_constraint=statdist,
                        count_mode=count_mode,
                        sparse=sparse, connectivity=connectivity,
                        dt_traj=dt_traj, maxiter=maxiter,
                        maxerr=maxerr, score_method=score_method, score_k=score_k,
                        mincount_connectivity=mincount_connectivity, core_set=core_set,
                        milestoning_method=milestoning_method)
        # estimate and return
        return mlmsm.estimate(dtrajs)
    elif weights == 'oom':
        if (statdist is not None) or (maxiter != 1000000) or (maxerr != 1e-8):
            import warnings
            warnings.warn("Values for statdist, maxiter or maxerr are ignored if OOM-correction is used.")
        _has_unassigned_states = any(-1 in d for d in dtrajs) if isinstance(dtrajs[0], _np.ndarray) else -1 in dtrajs
        if core_set is not None or _has_unassigned_states:
            raise NotImplementedError('Milestoning not implemented for OOMs.')
        oom_msm = _OOM_MSM(lag=lag, reversible=reversible, count_mode=count_mode,
                           sparse=sparse, connectivity=connectivity, dt_traj=dt_traj,
                           score_method=score_method, score_k=score_k,
                           mincount_connectivity=mincount_connectivity)
        # estimate and return
        return oom_msm.estimate(dtrajs)


def bayesian_markov_model(dtrajs, lag, reversible=True, statdist=None,
                          sparse=False, connectivity='largest',
                          count_mode='effective',
                          nsamples=100, conf=0.95, dt_traj='1 step',
                          show_progress=True, mincount_connectivity='1/n',
                          core_set=None, milestoning_method='last_core'):
    r""" Bayesian Markov model estimate using Gibbs sampling of the posterior

    Returns a :class:`BayesianMSM` that contains the
    estimated transition matrix and allows to compute a large number of
    quantities related to Markov models as well as their statistical
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
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.
    statdist : (M,) ndarray, optional
        Stationary vector on the full state-space. Transition matrix
        will be estimated such that statdist is its equilibrium
        distribution.
    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-tau` counts
          at time indexes

              .. math::

                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MSM.
        * 'sample' : A trajectory of length T will have :math:`T/tau` counts
          at time indexes

              .. math::

                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/tau)-1) \tau \rightarrow T)
    connectivity : str, optional, default = None
        Defines if the resulting HMM will be defined on all hidden states or on
        a connected subset. Connectivity is defined by counting only
        transitions with at least mincount_connectivity counts.
        If a subset of states is used, all estimated quantities (transition
        matrix, stationary distribution, etc) are only defined on this subset
        and are correspondingly smaller than nstates.
        Following modes are available:
        * None or 'all' : The active set is the full set of states.
          Estimation is done on all weakly connected subsets separately. The
          resulting transition matrix may be disconnected.
        * 'largest' : The active set is the largest reversibly connected set.
        * 'populous' : The active set is the reversibly connected set with
           most counts.
    nsample : int, optional, default=100
        number of transition matrix samples to compute and store
    conf : float, optional, default=0.95
        size of confidence intervals
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e.  there is no physical
        time unit. Specify by a number, whitespace and unit. Permitted units
        are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'
    show_progress : bool, default=True
        Show progressbars for calculation

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    core_set : None (default) or array like, dtype=int
        Definition of core set for milestoning MSMs.
        If set to None, replaces state -1 (if found in discrete trajectories) and
        performs milestone counting. No effect for Voronoi-discretized trajectories (default).
        If a list or np.ndarray is supplied, discrete trajectories will be assigned
        accordingly.

    milestoning_method : str
        Method to use for counting transitions in trajectories with unassigned frames.
        Currently available:
        |  'last_core',   assigns unassigned frames to last visited core

    Returns
    -------
    An :class:`BayesianMSM` object containing the Bayesian MSM estimator
    and the model.

    Example
    -------
    Note that the following example is only qualitatively and not
    quantitatively reproducible because it involves random numbers.

    We build a Bayesian Markov model for the following two trajectories at lag
    time 2:

    >>> from pyemma import msm
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]
    >>> mm = msm.bayesian_markov_model(dtrajs, 2, show_progress=False)

    The resulting Model is an MSM just like you get with estimate_markov_model
    Its transition matrix does also come from a maximum likelihood estimation,
    but it's slightly different from the estimate_markov_mode result because
    bayesian_markov_model uses an effective count matrix with statistically
    uncorrelated counts:

    >>> print(mm.transition_matrix)  # doctest: +SKIP
    [[ 0.70000001  0.16463699  0.135363  ]
     [ 0.38169055  0.          0.61830945]
     [ 0.12023989  0.23690297  0.64285714]]

    However bayesian_markov_model returns a SampledMSM object which is able to
    compute the probability distribution and statistical models of all methods
    that are offered by the MSM object. This works as follows. You can ask for
    the sample mean and specify the method you wanna evaluate as a string:

    >>> print(mm.sample_mean('transition_matrix'))  # doctest: +SKIP
    [[ 0.71108663  0.15947371  0.12943966]
     [ 0.41076105  0.          0.58923895]
     [ 0.13079372  0.23005443  0.63915185]]

    Likewise, the standard deviation by element:

    >>> print(mm.sample_std('transition_matrix'))  # doctest: +SKIP
    [[ 0.13707029  0.09479627  0.09200214]
     [ 0.15247454  0.          0.15247454]
     [ 0.07701315  0.09385258  0.1119089 ]]

    And this is the 95% (2 sigma) confidence interval. You can control the
    percentile using the conf argument in this function:

    >>> L, R = mm.sample_conf('transition_matrix')
    >>> print(L) # doctest: +SKIP
    >>> print(R)  # doctest: +SKIP
    [[ 0.44083423  0.03926518  0.0242113 ]
     [ 0.14102544  0.          0.30729828]
     [ 0.02440188  0.07629456  0.43682481]]
    [[ 0.93571706  0.37522581  0.40180041]
     [ 0.69307665  0.          0.8649215 ]
     [ 0.31029752  0.44035732  0.85994006]]

    If you wanna compute expectations of functions that require arguments,
    just pass these arguments as well:

    >>> print(mm.sample_std('mfpt', 0, 2)) # doctest: +SKIP
    12.9049811296

    And if you want to histogram the distribution or compute more complex
    statistical moment such as the covariance between different quantities,
    just get the full sample of your quantity of interest and evaluate it
    at will:

    >>> samples = mm.sample_f('mfpt', 0, 2)
    >>> print(samples[:4]) # doctest: +SKIP
    [7.9763615793248155, 8.6540958274695701, 26.295326015231058, 17.909895469938899]

    Internally, the SampledMSM object has 100 transition matrices (the number
    can be controlled by nsamples), that were computed by the transition matrix
    sampling method. All of the above sample functions iterate over these 100
    transition matrices and evaluate the requested function with the given
    parameters on each of them.


    .. autoclass:: pyemma.msm.estimators.bayesian_msm.BayesianMSM
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.bayesian_msm.BayesianMSM
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.bayesian_msm.BayesianMSM
            :attributes:

    References
    ----------
    .. [1] Trendelkamp-Schroer, B, H. Wu, F. Paul and F. Noe:
        Estimation and uncertainty of reversible Markov models.
        http://arxiv.org/abs/1507.05990

    """
    # TODO: store_data=True
    bmsm_estimator = _Bayes_MSM(lag=lag, reversible=reversible, statdist_constraint=statdist,
                                count_mode=count_mode, sparse=sparse, connectivity=connectivity,
                                dt_traj=dt_traj, nsamples=nsamples, conf=conf, show_progress=show_progress,
                                mincount_connectivity=mincount_connectivity,
                                core_set=core_set, milestoning_method=milestoning_method)
    return bmsm_estimator.estimate(dtrajs)


# =============================================================================
# HIDDEN MARKOV MODELS on discrete observation space
# =============================================================================


def timescales_hmsm(dtrajs, nstates, lags=None, nits=None, reversible=True, stationary=False,
                    connectivity=None, mincount_connectivity='1/n', separate=None, errors=None, nsamples=100,
                    stride=None, n_jobs=None, show_progress=True):
    r""" Calculate implied timescales from Hidden Markov state models estimated at a series of lag times.

    Warning: this can be slow!

    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories
    nstates : int
        number of hidden states
    lags : int, array-like with integers or None, optional
        integer lag times at which the implied timescales will be calculated. If set to None (default)
        as list of lag times will be automatically generated. For a single int, generate a set of lag times starting
        from 1 to lags, using a multiplier of 1.5 between successive lags.
    nits : int (optional)
        number of implied timescales to be computed. Will compute less if the
        number of states are smaller. None means the number of timescales will
        be determined automatically.
    connectivity : str, optional, default = None
        Defines if the resulting HMM will be defined on all hidden states or on
        a connected subset. Connectivity is defined by counting only
        transitions with at least mincount_connectivity counts.
        If a subset of states is used, all estimated quantities (transition
        matrix, stationary distribution, etc) are only defined on this subset
        and are correspondingly smaller than nstates.
        Following modes are available:
        * None or 'all' : The active set is the full set of states.
          Estimation is done on all weakly connected subsets separately. The
          resulting transition matrix may be disconnected.
        * 'largest' : The active set is the largest reversibly connected set.
        * 'populous' : The active set is the reversibly connected set with
           most counts.
    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    reversible : boolean (optional)
        Estimate transition matrix reversibly (True) or nonreversibly (False)
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently
        computed as the stationary distribution of the transition matrix. If False,
        it will be estimated from the starting states. Only set this to true if
        you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    errors : None | 'bayes'
        Specifies whether to compute statistical uncertainties (by default not),
        an which algorithm to use if yes. The only option is currently 'bayes'.
        This algorithm is much faster than MSM-based error calculation because
        the involved matrices are much smaller.
    nsamples : int
        Number of approximately independent HMSM samples generated for each lag
        time for uncertainty quantification. Only used if errors is not None.
    n_jobs : int
        how many subprocesses to start to estimate the models for each lag time.
    show_progress : bool, default=True
        Show progressbars for calculation?

    Returns
    -------
    itsobj : :class:`ImpliedTimescales <pyemma.msm.ImpliedTimescales>` object

    See also
    --------
    ImpliedTimescales
        The object returned by this function.
    pyemma.plots.plot_implied_timescales
        Plotting function for the :class:`ImpliedTimescales <pyemma.msm.ImpliedTimescales>` object

    Example
    -------
    >>> from pyemma import msm
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)
    >>> dtraj = [0,1,1,0,0,0,1,1,0,0,0,1,2,2,2,2,2,2,2,2,2,1,1,0,0,0,1,1,0,1,0]   # mini-trajectory
    >>> ts = msm.timescales_hmsm(dtraj, 2, [1,2,3,4], show_progress=False)
    >>> print(ts.timescales) # doctest: +ELLIPSIS
    [[ 5.786]
     [ 5.143]
     [ 4.44 ]
     [ 3.677]]

    .. autoclass:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.implied_timescales.ImpliedTimescales
            :attributes:

    References
    ----------
    Implied timescales as a lagtime-selection and MSM-validation approach were
    suggested in [1]_. Hidden Markov state model estimation is done here as
    described in [2]_. For uncertainty quantification we employ the Bayesian
    sampling algorithm described in [3]_.

    .. [1] Swope, W. C. and J. W. Pitera and F. Suits: Describing protein
        folding kinetics by molecular dynamics simulations:  1. Theory.
        J. Phys. Chem. B 108: 6571-6581 (2004)

    .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
        Markov models for calculating kinetics and metastable states of
        complex molecules. J. Chem. Phys. 139, 184114 (2013)

    .. [3] J. D. Chodera et al:
        Bayesian hidden Markov model analysis of single-molecule force
        spectroscopy: Characterizing kinetics under measurement uncertainty
        arXiv:1108.1430 (2011)

    """
    # format data
    dtrajs = _types.ensure_dtraj_list(dtrajs)

    # MLE or error estimation?
    if errors is None:
        if stride is None:
            stride = 1
        estimator = _ML_HMSM(nstates=nstates, reversible=reversible, stationary=stationary, connectivity=connectivity,
                             stride=stride, mincount_connectivity=mincount_connectivity, separate=separate)
    elif errors == 'bayes':
        if stride is None:
            stride = 'effective'
        estimator = _Bayes_HMSM(nstates=nstates, reversible=reversible, stationary=stationary,
                                connectivity=connectivity, mincount_connectivity=mincount_connectivity,
                                stride=stride, separate=separate, show_progress=show_progress, nsamples=nsamples)
    else:
        raise NotImplementedError('Error estimation method'+str(errors)+'currently not implemented')

    # go
    itsobj = _ImpliedTimescales(estimator, lags=lags, nits=nits, n_jobs=n_jobs,
                                show_progress=show_progress)
    itsobj.estimate(dtrajs)
    return itsobj


def estimate_hidden_markov_model(dtrajs, nstates, lag, reversible=True, stationary=False,
                                 connectivity=None, mincount_connectivity='1/n', separate=None, observe_nonempty=True,
                                 stride=1, dt_traj='1 step', accuracy=1e-3, maxit=1000):
    r""" Estimates a Hidden Markov state model from discrete trajectories

    Returns a :class:`MaximumLikelihoodHMSM` that contains a transition
    matrix between a few (hidden) metastable states. Each metastable state has
    a probability distribution of visiting the discrete 'microstates' contained
    in the input trajectories. The resulting object is a hidden Markov model
    that allows to compute a large number of quantities.

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
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently
        computed as the stationary distribution of the transition matrix. If False,
        it will be estimated from the starting states. Only set this to true if
        you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    connectivity : str, optional, default = None
        Defines if the resulting HMM will be defined on all hidden states or on
        a connected subset. Connectivity is defined by counting only
        transitions with at least mincount_connectivity counts.
        If a subset of states is used, all estimated quantities (transition
        matrix, stationary distribution, etc) are only defined on this subset
        and are correspondingly smaller than nstates.
        Following modes are available:
        * None or 'all' : The active set is the full set of states.
          Estimation is done on all weakly connected subsets separately. The
          resulting transition matrix may be disconnected.
        * 'largest' : The active set is the largest reversibly connected set.
        * 'populous' : The active set is the reversibly connected set with
           most counts.
    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    observe_nonempty : bool
        If True, will restricted the observed states to the states that have
        at least one observation in the lagged input trajectories.
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
    hmsm : :class:`MaximumLikelihoodHMSM <pyemma.msm.MaximumLikelihoodHMSM>`
        Estimator object containing the HMSM and estimation information.

    Example
    -------
    >>> from pyemma import msm
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.estimate_hidden_markov_model(dtrajs, 2, 2)

    We have estimated a 2x2 hidden transition matrix between the metastable
    states:

    >>> print(mm.transition_matrix)
    [[ 0.684  0.316]
     [ 0.242  0.758]]

    With the equilibrium distribution:

    >>> print(mm.stationary_distribution) # doctest: +ELLIPSIS
    [ 0.43...  0.56...]

    The observed states are the three discrete clusters that we have in our
    discrete trajectory:

    >>> print(mm.observable_set)
    [0 1 2]

    The metastable distributions (mm.metastable_distributions), or equivalently
    the observation probabilities are the probability to be in a given cluster
    ('microstate') if we are in one of the hidden metastable states.
    So it's a 2 x 3 matrix:

    >>> print(mm.observation_probabilities) # doctest: +SKIP
    [[ 0.9620883   0.0379117   0.        ]
     [ 0.          0.28014352  0.71985648]]

    The first metastable state ist mostly in cluster 0, and a little bit in the
    transition state cluster 1. The second metastable state is less well
    defined, but mostly in cluster 2 and less prominently in the transition
    state cluster 1.

    We can print the lifetimes of the metastable states:

    >>> print(mm.lifetimes) # doctest: +ELLIPSIS
    [ 5...  7...]

    And the timescale of the hidden transition matrix - now we only have one
    relaxation timescale:

    >>> print(mm.timescales())  # doctest: +ELLIPSIS
    [ 2.4...]

    The mean first passage times can also be computed between metastable states:

    >>> print(mm.mfpt(0, 1))  # doctest: +ELLIPSIS
    6.3...


    .. autoclass:: pyemma.msm.estimators.maximum_likelihood_hmsm.MaximumLikelihoodHMSM
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_hmsm.MaximumLikelihoodHMSM
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_hmsm.MaximumLikelihoodHMSM
            :attributes:


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


    """
    # initialize HMSM estimator
    hmsm_estimator = _ML_HMSM(lag=lag, nstates=nstates, reversible=reversible, stationary=stationary,
                              msm_init='largest-strong',
                              connectivity=connectivity, mincount_connectivity=mincount_connectivity, separate=separate,
                              observe_nonempty=observe_nonempty, stride=stride, dt_traj=dt_traj,
                              accuracy=accuracy, maxit=maxit)
    # run estimation
    return hmsm_estimator.estimate(dtrajs)


def bayesian_hidden_markov_model(dtrajs, nstates, lag, nsamples=100, reversible=True, stationary=False,
                                 connectivity=None, mincount_connectivity='1/n', separate=None, observe_nonempty=True,
                                 stride='effective', conf=0.95, dt_traj='1 step', store_hidden=False, show_progress=True):
    r""" Bayesian Hidden Markov model estimate using Gibbs sampling of the posterior

    Returns a :class:`BayesianHMSM` that contains
    the estimated hidden Markov model [1]_ and a Bayesian estimate [2]_ that
    contains samples around this estimate to estimate uncertainties.

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
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently
        computed as the stationary distribution of the transition matrix. If False,
        it will be estimated from the starting states. Only set this to true if
        you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    connectivity : str, optional, default = None
        Defines if the resulting HMM will be defined on all hidden states or on
        a connected subset. Connectivity is defined by counting only
        transitions with at least mincount_connectivity counts.
        If a subset of states is used, all estimated quantities (transition
        matrix, stationary distribution, etc) are only defined on this subset
        and are correspondingly smaller than nstates.
        Following modes are available:
        * None or 'all' : The active set is the full set of states.
          Estimation is done on all weakly connected subsets separately. The
          resulting transition matrix may be disconnected.
        * 'largest' : The active set is the largest reversibly connected set.
        * 'populous' : The active set is the reversibly connected set with
           most counts.
    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    observe_nonempty : bool
        If True, will restricted the observed states to the states that have
        at least one observation in the lagged input trajectories.
    nsamples : int, optional, default=100
        number of transition matrix samples to compute and store
    stride : str or int, default='effective'
        stride between two lagged trajectories extracted from the input
        trajectories. Given trajectory s[t], stride and lag will result
        in trajectories
            s[0], s[tau], s[2 tau], ...
            s[stride], s[stride + tau], s[stride + 2 tau], ...
        Setting stride = 1 will result in using all data (useful for
        maximum likelihood estimator), while a Bayesian estimator requires
        a longer stride in order to have statistically uncorrelated
        trajectories. Setting stride = None 'effective' uses the largest
        neglected timescale as an estimate for the correlation time and
        sets the stride accordingly.
    conf : float, optional, default=0.95
        size of confidence intervals
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step. May be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e.  there is no physical
        time unit. Specify by a number, whitespace and unit. Permitted units
        are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs
    show_progress : bool, default=True
        Show progressbars for calculation?

    Returns
    -------
    An :class:`BayesianHMSM` object containing a
    transition matrix and various other HMM-related quantities and statistical
    uncertainties.

    Example
    -------
    Note that the following example is only qualitative and not
    quantitatively reproducible because random numbers are involved

    >>> from pyemma import msm
    >>> dtrajs = [[0,1,2,2,2,2,1,2,2,2,1,0,0,0,0,0,0,0], [0,0,0,0,1,1,2,2,2,2,2,2,2,1,0,0]]  # two trajectories
    >>> mm = msm.bayesian_hidden_markov_model(dtrajs, 2, 2, show_progress=False)

    We compute the stationary distribution (here given by the maximum
    likelihood estimate), and the 1-sigma uncertainty interval. You can see
    that the uncertainties are quite large (we have seen only very few
    transitions between the metastable states:

    >>> pi = mm.stationary_distribution
    >>> piL,piR = mm.sample_conf('stationary_distribution')
    >>> for i in range(2): print(pi[i],' -',piL[i],'+',piR[i])  # doctest: +SKIP
    0.459176653019  - 0.268314552886 + 0.715326151685
    0.540823346981  - 0.284761476984 + 0.731730375713

    Let's look at the lifetimes of metastable states. Now we have really huge
    uncertainties. In states where one state is more probable than the other,
    the mean first passage time from the more probable to the less probable
    state is much higher than the reverse:

    >>> l = mm.lifetimes
    >>> lL, lR = mm.sample_conf('lifetimes')
    >>> for i in range(2): print(l[i],' -',lL[i],'+',lR[i])  # doctest: +SKIP
    7.18543434854  - 6.03617757784 + 80.1298222741
    8.65699332061  - 5.35089540896 + 30.1719505772

    In contrast the relaxation timescale is less uncertain. This is because
    for a two-state system the relaxation timescale is dominated by the faster
    passage, which is less uncertain than the slower passage time:

    >>> ts = mm.timescales()
    >>> tsL,tsR = mm.sample_conf('timescales')
    >>> print(ts[0],' -',tsL[0],'+',tsR[0])  # doctest: +SKIP
    3.35310468086  - 2.24574587978 + 8.34383177258


    .. autoclass:: pyemma.msm.estimators.bayesian_hmsm.BayesianHMSM
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.bayesian_hmsm.BayesianHMSM
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.bayesian_hmsm.BayesianHMSM
            :attributes:

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
        Markov models for calculating kinetics and metastable states of complex
        molecules. J. Chem. Phys. 139, 184114 (2013)
    .. [2] J. D. Chodera Et Al: Bayesian hidden Markov model analysis of
        single-molecule force spectroscopy: Characterizing kinetics under
        measurement uncertainty. arXiv:1108.1430 (2011)

    """
    bhmsm_estimator = _Bayes_HMSM(lag=lag, nstates=nstates, stride=stride, nsamples=nsamples, reversible=reversible,
                                  stationary=stationary,
                                  connectivity=connectivity, mincount_connectivity=mincount_connectivity,
                                  separate=separate, observe_nonempty=observe_nonempty,
                                  dt_traj=dt_traj, conf=conf, store_hidden=store_hidden, show_progress=show_progress)
    return bhmsm_estimator.estimate(dtrajs)

def estimate_augmented_markov_model(dtrajs, ftrajs, lag, m, sigmas,
                          count_mode='sliding',  connectivity='largest',
                          dt_traj='1 step', maxiter=1000000, eps=0.05, maxcache=3000,
                          core_set=None, milestoning_method='last_core'):
    r""" Estimates an Augmented Markov model from discrete trajectories and experimental data

    Returns a :class:`AugmentedMarkovModel` that
    contains the estimated transition matrix and allows to compute a
    large number of quantities related to Markov models.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    ftrajs : list of trajectories of microscopic observables. Has to have
        the same shape (number of trajectories and timesteps) as dtrajs.
        Each timestep in each trajectory should match the shape of m and sigma, k.
    lag : int
        lag time at which transitions are counted and the transition matrix is
        estimated.
    m   :  ndarray(k)
        Experimental averages.
    sigmas : ndarray(k)
        Standard error for each experimental observable.
    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

              .. math::

                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MSM.
        * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
          at time indexes

              .. math::

                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)
    connectivity : str, optional
        Connectivity mode. Three methods are intended (currently only
        'largest' is implemented)

        * 'largest' : The active set is the largest reversibly
          connected set. All estimation will be done on this subset
          and all quantities (transition matrix, stationary
          distribution, etc) are only defined on this subset and are
          correspondingly smaller than the full set of states

        * 'all' : The active set is the full set of states. Estimation
          will be conducted on each reversibly connected set
          separately. That means the transition matrix will decompose
          into disconnected submatrices, the stationary vector is only
          defined within subsets, etc. Currently not implemented.

        * 'none' : The active set is the full set of
          states. Estimation will be conducted on the full set of
          states without ensuring connectivity. This only permits
          nonreversible estimation. Currently not implemented.
    dt_traj : str, optional
        Description of the physical time corresponding to the lag. May
        be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e. there is no
        physical time unit.  Specify by a number, whitespace and
        unit. Permitted units are (* is an arbitrary string):

        *  'fs',  'femtosecond*'
        *  'ps',  'picosecond*'
        *  'ns',  'nanosecond*'
        *  'us',  'microsecond*'
        *  'ms',  'millisecond*'
        *  's',   'second*'

    maxiter : int, optional
        Optional parameter with specifies the maximum number of
        updates for Lagrange multiplier estimation.

    eps : float, optional
        Additional convergence criterion used when some experimental data
        are outside the support of the simulation. The value of the eps
        parameter is the threshold of the relative change in the predicted
        observables as a function of fixed-point iteration:
        $$ \mathrm{eps} > \frac{\mid o_{\mathrm{pred}}^{(i+1)}-o_{\mathrm{pred}}^{(i)}\mid }{\sigma}. $$

    maxcache : int, optional
        Parameter which specifies the maximum size of cache used
        when performing estimation of AMM, in megabytes.

    core_set : None (default) or array like, dtype=int
        Definition of core set for milestoning MSMs.
        If set to None, replaces state -1 (if found in discrete trajectories) and
        performs milestone counting. No effect for Voronoi-discretized trajectories (default).
        If a list or np.ndarray is supplied, discrete trajectories will be assigned
        accordingly.

    milestoning_method : str
        Method to use for counting transitions in trajectories with unassigned frames.
        Currently available:
        |  'last_core',   assigns unassigned frames to last visited core

    Returns
    -------
    amm : :class:`AugmentedMarkovModel <pyemma.msm.AugmentedMarkovModel>`
        Estimator object containing the AMM and estimation information.

    See also
    --------
    AugmentedMarkovModel
        An AMM object that has been estimated from data


    .. autoclass:: pyemma.msm.estimators.maximum_likelihood_msm.AugmentedMarkovModel
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_msm.AugmentedMarkovModel
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.estimators.maximum_likelihood_msm.AugmentedMarkovModel
            :attributes:


    References
    ----------
    .. [1] Olsson S, Wu H, Paul F, Clementi C, Noe F "Combining experimental and simulation data
       of molecular processes via augmented Markov models" PNAS (2017), 114(31), pp. 8265-8270
       doi: 10.1073/pnas.1704803114

    """
    # check input
    if _np.all(sigmas>0):
        _w = 1./(2*sigmas**2.)
    else:
        raise ValueError('Zero or negative standard errors supplied. Please revise input')
    if ftrajs[0].ndim < 2:
        raise ValueError("Supplied feature trajectories have inappropriate dimensions (%d) should be atleast 2."%ftrajs[0].ndim)
    if len(dtrajs) != len(ftrajs):
        raise ValueError("A different number of dtrajs and ftrajs were supplied as input. They must have exactly a one-to-one correspondence.")
    elif not _np.all([len(dt)==len(ft) for dt,ft in zip(dtrajs, ftrajs)]):
        raise ValueError("One or more supplied dtraj-ftraj pairs do not have the same length.")
    else:
        # MAKE E matrix
        dta = _np.concatenate(dtrajs)
        fta = _np.concatenate(ftrajs)
        if core_set is not None:
            all_markov_states = set(core_set)
        else:
            all_markov_states = set(dta)
        _E = _np.zeros((len(all_markov_states), fta.shape[1]))
        for i, s in enumerate(all_markov_states):
            _E[i, :] = fta[_np.where(dta == s)].mean(axis=0)
        # transition matrix estimator
        mlamm = _ML_AMM(lag=lag, count_mode=count_mode,
                        connectivity=connectivity,
                        dt_traj=dt_traj, maxiter=maxiter, max_cache=maxcache,
                        E=_E, w=_w, m=m, core_set=core_set,
                        milestoning_method=milestoning_method)
        # estimate and return
        return mlamm.estimate(dtrajs)

def tpt(msmobj, A, B):
    r""" A->B reactive flux from transition path theory (TPT)

    The returned :class:`ReactiveFlux <pyemma.msm.models.ReactiveFlux>` object
    can be used to extract various quantities of the flux, as well as to
    compute A -> B transition pathways, their weights, and to coarse-grain
    the flux onto sets of states.

    Parameters
    ----------
    msmobj : :class:`MSM <pyemma.msm.MSM>` object
        Markov state model (MSM) object
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B

    Returns
    -------
    tptobj : :class:`ReactiveFlux <pyemma.msm.models.ReactiveFlux>` object
        An object containing the reactive A->B flux network
        and several additional quantities, such as the stationary probability,
        committors and set definitions.

    See also
    --------
    :class:`ReactiveFlux <pyemma.msm.models.ReactiveFlux>`
        Reactive Flux model


    .. autoclass:: pyemma.msm.models.ReactiveFlux
        :members:
        :undoc-members:

        .. rubric:: Methods

        .. autoautosummary:: pyemma.msm.models.ReactiveFlux
           :methods:

        .. rubric:: Attributes

        .. autoautosummary:: pyemma.msm.models.ReactiveFlux
            :attributes:

    References
    ----------
    Transition path theory was introduced for space-continuous dynamical
    processes, such as Langevin dynamics, in [1]_, [2]_ introduces discrete
    transition path theory for Markov jump processes (Master equation models,
    rate matrices) and pathway decomposition algorithms. [3]_ introduces
    transition path theory for Markov state models (MSMs) and some analysis
    algorithms. In this function, the equations described in [3]_ are applied.

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


    Computes the A->B reactive flux using transition path theory (TPT)

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix (default) or Rate matrix (if rate_matrix=True)
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction
    rate_matrix = False : boolean
        By default (False), T is a transition matrix.
        If set to True, T is a rate matrix.

    Notes
    -----
    The central object used in transition path theory is
    the forward and backward comittor function.

    TPT (originally introduced in [1]) for continous systems has a
    discrete version outlined in [2]. Here, we use the transition
    matrix formulation described in [3].

    See also
    --------
    msmtools.analysis.committor, ReactiveFlux

    References
    ----------
    .. [1] W. E and E. Vanden-Eijnden.
        Towards a theory of transition paths.
        J. Stat. Phys. 123: 503-523 (2006)
    .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)
    .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and T. Weikl:
        Constructing the Full Ensemble of Folding Pathways from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)

    """
    from msmtools.flux import flux_matrix, to_netflux
    import msmtools.analysis as msmana

    T = msmobj.transition_matrix
    mu = msmobj.stationary_distribution
    A = _types.ensure_ndarray(A, kind='i')
    B = _types.ensure_ndarray(B, kind='i')

    if len(A) == 0 or len(B) == 0:
        raise ValueError('set A or B is empty')
    n = T.shape[0]
    if len(A) > n or len(B) > n or max(A) > n or max(B) > n:
        raise ValueError('set A or B defines more states, than given transition matrix.')

    # forward committor
    qplus = msmana.committor(T, A, B, forward=True)
    # backward committor
    if msmana.is_reversible(T, mu=mu):
        qminus = 1.0 - qplus
    else:
        qminus = msmana.committor(T, A, B, forward=False, mu=mu)
    # gross flux
    grossflux = flux_matrix(T, mu, qminus, qplus, netflux=False)
    # net flux
    netflux = to_netflux(grossflux)

    # construct flux object
    from .models.reactive_flux import ReactiveFlux

    F = ReactiveFlux(A, B, netflux, mu=mu, qminus=qminus, qplus=qplus, gross_flux=grossflux, dt_model=msmobj.dt_model)
    return F
