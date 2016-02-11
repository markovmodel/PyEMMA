
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

from __future__ import absolute_import
from six.moves import range

import numpy as np
import msmtools.estimation as msmest
from pyemma.msm.estimators.estimated_hmsm import EstimatedHMSM as _EstimatedHMSM

from pyemma.msm.estimators.estimated_msm import EstimatedMSM as _EstimatedMSM
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit


# TODO: This parameter was in the docstring but is not used:
#     store_data : bool
#         True: estimate() returns an :class:`pyemma.msm.EstimatedMSM` object
#         with discrete trajectories and counts stored. False: estimate() returns
#         a plain :class:`pyemma.msm.MSM` object that only contains the
#         transition matrix and quantities derived from it.
# TODO: currently, it's not possible to start with disconnected matrices.


class MaximumLikelihoodHMSM(_Estimator, _EstimatedHMSM):
    r"""Maximum likelihood estimator for a Hidden MSM given a MSM"""

    def __init__(self, nstates=2, lag=1, stride=1, msm_init='largest-strong', reversible=True, stationary=False,
                 connectivity=None, mincount_connectivity='1/n',
                 separate=None, dt_traj='1 step', accuracy=1e-3, maxit=1000):
        r"""Maximum likelihood estimator for a Hidden MSM given a MSM

        Parameters
        ----------
        nstates : int, optional, default=2
            number of hidden states
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        stride : str or int, default=1
            stride between two lagged trajectories extracted from the input
            trajectories. Given trajectory s[t], stride and lag will result
            in trajectories
                s[0], s[lag], s[2 lag], ...
                s[stride], s[stride + lag], s[stride + 2 lag], ...
            Setting stride = 1 will result in using all data (useful for maximum
            likelihood estimator), while a Bayesian estimator requires a longer
            stride in order to have statistically uncorrelated trajectories.
            Setting stride = 'effective' uses the largest neglected timescale as
            an estimate for the correlation time and sets the stride accordingly
        msm_init : str or :class:`MSM <pyemma.msm.estimators.msm_estimated.MSM>`
            MSM object to initialize the estimation, or one of following keywords:
            * 'largest-strong' | None (default) : Estimate MSM on the largest
                strongly connected set and use spectral clustering to generate an
                initial HMM
            * 'all' : Estimate MSM(s) on the full state space to initialize the
                HMM. This estimate maybe weakly connected or disconnected.
        reversible : bool, optional, default = True
            If true compute reversible MSM, else non-reversible MSM
        stationary : bool, optional, default=False
            If True, the initial distribution of hidden states is self-consistently computed as the stationary
            distribution of the transition matrix. If False, it will be estimated from the starting states.
            Only set this to true if you're sure that the observation trajectories are initiated from a global
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
        dt_traj : str, optional, default='1 step'
            Description of the physical time corresponding to the trajectory time
            step.  May be used by analysis algorithms such as plotting tools to
            pretty-print the axes. By default '1 step', i.e. there is no physical
            time unit. Specify by a number, whitespace and unit. Permitted units
            are (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        accuracy : float, optional, default = 1e-3
            convergence threshold for EM iteration. When two the likelihood does
            not increase by more than accuracy, the iteration is stopped
            successfully.
        maxit : int, optional, default = 1000
            stopping criterion for EM iteration. When so many iterations are
            performed without reaching the requested accuracy, the iteration is
            stopped without convergence (a warning is given)

        """
        self.nstates = nstates
        self.lag = lag
        self.stride = stride
        self.msm_init = msm_init
        self.reversible = reversible
        self.stationary = stationary
        self.connectivity = connectivity
        if mincount_connectivity == '1/n':
            mincount_connectivity = 1.0/float(nstates)
        self.mincount_connectivity = mincount_connectivity
        self.separate = separate
        self.dt_traj = dt_traj
        self.timestep_traj = TimeUnit(dt_traj)
        self.accuracy = accuracy
        self.maxit = maxit


    #TODO: store_data is mentioned but not implemented or used!
    def _estimate(self, dtrajs):
        """

        Parameters
        ----------

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.estimators.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        import bhmm
        # ensure right format
        dtrajs = _types.ensure_dtraj_list(dtrajs)

        # CHECK LAG
        trajlengths = [np.size(dtraj) for dtraj in dtrajs]
        if self.lag >= np.max(trajlengths):
            raise ValueError('Illegal lag time ' + str(self.lag) + ' exceeds longest trajectory length')
        if self.lag > np.mean(trajlengths):
            self.logger.warning('Lag time ' + str(self.lag) + ' is on the order of mean trajectory length'
                                + np.mean(trajlengths) + '. It is recommended to fit four lag times in each '
                                + 'trajectory. HMM might be inaccurate.')

        # EVALUATE STRIDE
        if self.stride == 'effective':
            # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
            # how many uncorrelated counts we can make
            self.stride = self.lag
            # get a quick estimate from the spectral radius of the nonreversible
            from pyemma.msm import estimate_markov_model
            msm_nr = estimate_markov_model(dtrajs, lag=self.lag, reversible=False, sparse=False,
                                           connectivity='largest', dt_traj=self.timestep_traj)
            # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
            # estimate of the decorrelation time
            if msm_nr.nstates > self.nstates:
                corrtime = max(1, msm_nr.timescales()[self.nstates-1])
                # use the smaller of these two pessimistic estimates
                self.stride = int(min(self.lag, 2*corrtime))

        # LAG AND STRIDE DATA
        dtrajs_lagged_strided = bhmm.lag_observations(dtrajs, self.lag, stride=self.stride)

        # OBSERVATION SET
        observe_subset = None

        # INIT HMM
        from bhmm import init_discrete_hmm
        if self.msm_init=='largest-strong':
            hmm_init = init_discrete_hmm(dtrajs_lagged_strided, self.nstates, lag=1,
                                         reversible=self.reversible, stationary=True, regularize=True,
                                         method='lcs-spectral', separate=self.separate)
        elif self.msm_init=='all':
            hmm_init = init_discrete_hmm(dtrajs_lagged_strided, self.nstates, lag=1,
                                         reversible=self.reversible, stationary=True, regularize=True,
                                         method='spectral', separate=self.separate)
        elif isinstance(self.msm_init, _EstimatedMSM):  # initial MSM given.
            from bhmm.init.discrete import init_discrete_hmm_spectral
            p0, P0, pobs0 = init_discrete_hmm_spectral(self.msm_init.count_matrix_full, self.nstates,
                                                       reversible=self.reversible, stationary=True,
                                                       active_set=self.msm_init.active_set,
                                                       P=self.msm_init.transition_matrix, separate=self.separate)
            hmm_init = bhmm.discrete_hmm(p0, P0, pobs0)
            observe_subset = self.msm_init.active_set
        else:
            raise ValueError('Unknown MSM initialization option: ' + str(self.msm_init))

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------

        # run EM
        from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as _MaximumLikelihoodEstimator
        hmm_est = _MaximumLikelihoodEstimator(dtrajs_lagged_strided, self.nstates, initial_model=hmm_init,
                                              output='discrete', reversible=self.reversible, stationary=self.stationary,
                                              accuracy=self.accuracy, maxit=self.maxit)
        # run
        hmm_est.fit()
        # package in discrete HMM
        self.hmm = bhmm.DiscreteHMM(hmm_est.hmm)

        # get model parameters
        self.initial_distribution = self.hmm.initial_distribution
        transition_matrix = self.hmm.transition_matrix
        observation_probabilities = self.hmm.output_probabilities

        # get estimation parameters
        self.likelihoods = hmm_est.likelihoods  # Likelihood history
        self.likelihood = self.likelihoods[-1]
        self.hidden_state_probabilities = hmm_est.hidden_state_probabilities  # gamma variables
        self.hidden_state_trajectories = hmm_est.hmm.hidden_state_trajectories  # Viterbi path
        self.count_matrix = hmm_est.count_matrix  # hidden count matrix
        self.initial_count = hmm_est.initial_count  # hidden init count
        self._active_set = np.arange(self.nstates)

        # TODO: it can happen that we loose states due to striding. Should we lift the output probabilities afterwards?
        # parametrize self
        self._dtrajs_full = dtrajs
        self._dtrajs_lagged = dtrajs_lagged_strided
        self._observable_set = np.arange(msmest.number_of_states(dtrajs_lagged_strided))
        self._dtrajs_obs = dtrajs
        self.set_model_params(P=transition_matrix, pobs=observation_probabilities,
                              reversible=self.reversible, dt_model=self.timestep_traj.get_scaled(self.lag))

        # TODO: perhaps remove connectivity and just rely on .submodel()?
        # deal with connectivity
        states_subset = None
        if self.connectivity == 'largest':
            states_subset = 'largest-strong'
        elif self.connectivity == 'populous':
            states_subset = 'populous-strong'

        # return submodel (will return self if all None)
        return self.submodel(states=states_subset, obs=observe_subset, mincount_connectivity=self.mincount_connectivity)

    def cktest(self, mlags=10, conf=0.95, err_est=False, show_progress=True):
        """ Conducts a Chapman-Kolmogorow test.

        Parameters
        ----------
        mlags : int or int-array, default=10
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10). The setting None will choose mlags automatically
            according to the longest available trajectory
        conf : float, optional, default = 0.95
            confidence interval
        err_est : bool, default=False
            compute errors also for all estimations (computationally expensive)
            If False, only the prediction will get error bars, which is often
            sufficient to validate a model.
        show_progress : bool, default=True
            Show progressbars for calculation?

        Returns
        -------
        cktest : :class:`ChapmanKolmogorovValidator <pyemma.msm.ChapmanKolmogorovValidator>`

        References
        ----------
        This is an adaption of the Chapman-Kolmogorov Test described in detail
        in [1]_ to Hidden MSMs as described in [2]_.

        .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
            Chodera, C Schuette and F Noe. 2011. Markov models of
            molecular kinetics: Generation and validation. J Chem Phys
            134: 174105

        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of complex
            molecules. J. Chem. Phys. 139, 184114 (2013)

        """
        from pyemma.msm.estimators import ChapmanKolmogorovValidator
        ck = ChapmanKolmogorovValidator(self, self, np.eye(self.nstates),
                                        mlags=mlags, conf=conf, err_est=err_est,
                                        show_progress=show_progress)
        ck.estimate(self._dtrajs_full)
        return ck