
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
from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MSMEstimator
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

    def __init__(self, nstates=2, lag=1, stride=1, msm_init=None, reversible=True, stationary=False,
                 connectivity=None, mincount_connectivity='1/n', observe_active=False,
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
        msm_init : :class:`MSM <pyemma.msm.estimators.msm_estimated.MSM>`
            MSM object to initialize the estimation
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
        observe_active : bool, optional, default=False
            True: Restricts the observation set to the active states of the initial MSM.
            False: All states are in the observation set.
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
        self.observe_active = observe_active
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

        # check lag
        trajlengths = [np.size(dtraj) for dtraj in dtrajs]
        if self.lag >= np.max(trajlengths):
            raise ValueError('Illegal lag time ' + str(self.lag) + ' exceeds longest trajectory length')
        if self.lag > np.mean(trajlengths):
            self.logger.warning('Lag time ' + str(self.lag) + ' is on the order of mean trajectory length'
                                + np.mean(trajlengths) + '. It is recommended to fit four lag times in each '
                                + 'trajectory. HMM might be inaccurate.')

        # if no initial MSM is given, estimate it now
        if self.msm_init is None:
            # estimate with sparse=False, because we need to do PCCA which is currently not implemented for sparse
            # estimate with connectivity='largest', because otherwise we might have a lot of singlet states
            msm_estimator = _MSMEstimator(lag=self.lag, reversible=self.reversible, sparse=False,
                                          connectivity='largest', dt_traj=self.timestep_traj)
            msm_init = msm_estimator.estimate(dtrajs)
        else:
            assert isinstance(self.msm_init, _EstimatedMSM), 'msm_init must be of type EstimatedMSM'
            msm_init = self.msm_init
            self.reversible = msm_init.is_reversible

        # print 'Connected set: ', msm_init.active_set

        # generate lagged observations
        if self.stride == 'effective':
            # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
            # how many uncorrelated counts we can make
            self.stride = self.lag
            # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
            # estimate of the decorrelation time
            if msm_init.nstates > self.nstates:
                corrtime = int(max(1, msm_init.timescales()[self.nstates-1]))
                # use the smaller of these two pessimistic estimates
                self.stride = min(self.stride, 2*corrtime)
        # TODO: Here we always use the full observation state space for the estimation.
        dtrajs_lagged = bhmm.lag_observations(dtrajs, self.lag, stride=self.stride)

        # check input
        assert _types.is_int(self.nstates) and self.nstates > 1 and self.nstates <= msm_init.nstates, \
            'nstates must be an int in [2,msmobj.nstates]'
        # if hmm.nstates = msm.nstates there is no problem. Otherwise, check spectral gap
        if msm_init.nstates > self.nstates:
            timescale_ratios = msm_init.timescales()[:-1] / msm_init.timescales()[1:]
            if timescale_ratios[self.nstates-2] < 1.5:
                self.logger.warning('Requested coarse-grained model with ' + str(self.nstates) + ' metastable states at ' +
                                 'lag=' + str(self.lag) + '.' + 'The ratio of relaxation timescales between ' +
                                 str(self.nstates) + ' and ' + str(self.nstates+1) + ' states is only ' +
                                 str(timescale_ratios[self.nstates-2]) + ' while we recommend at least 1.5. ' +
                                 ' It is possible that the resulting HMM is inaccurate. Handle with caution.')

        # set things from MSM
        # TODO: dtrajs_obs is set here, but not used in estimation. Estimation is always done with
        # TODO: respect to full observation (see above). This is confusing. Define how we want to do this in gen.
        # TODO: observable set is also not used, it is just saved.
        nstates_obs_full = msm_init.nstates_full
        if self.observe_active:
            nstates_obs = msm_init.nstates
            observable_set = msm_init.active_set
            dtrajs_obs = msm_init.discrete_trajectories_active
        else:
            nstates_obs = msm_init.nstates_full
            observable_set = np.arange(nstates_obs_full)
            dtrajs_obs = msm_init.discrete_trajectories_full

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------

        # TODO: this is really not observe_active, but this chooses how to initialize the HMM.
        # TODO: Do we need a separate flag here? And if we estimate disconnected (here observe_active=True),
        # TODO: Why do we even need the initial MSM? It's just a waste of time in that case.
        if self.observe_active:
            Cinit = msm_init.count_matrix_full
            Pinit = msm_init.transition_matrix
            # TODO: active_set confusing name. msm_active_set or similar?
            active_set = msm_init.active_set
        else:
            Cinit = msm_init.count_matrix_full
            # make sure we're strongly connected
            Cinit += msmest.prior_neighbor(Cinit, 0.001)
            nonempty = np.where(Cinit.sum(axis=0) + Cinit.sum(axis=1) > 0)[0]
            Cinit[nonempty, nonempty] = np.maximum(Cinit[nonempty, nonempty], 0.001)
            Pinit = None
            active_set = None

        from bhmm.init.discrete import estimate_initial_hmm
        hmm_init = estimate_initial_hmm(Cinit, self.nstates, reversible=self.reversible,
                                        active_set=active_set, P=Pinit, separate=self.separate)

        # run EM
        from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as _MaximumLikelihoodEstimator
        hmm_est = _MaximumLikelihoodEstimator(dtrajs_lagged, self.nstates, initial_model=hmm_init, type='discrete',
                                              reversible=self.reversible, stationary=self.stationary,
                                              accuracy=self.accuracy, maxit=self.maxit,
                                              mincount_connectivity=self.mincount_connectivity)
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

        # deal with connectivity
        if self.connectivity == 'largest':
            from msmtools.estimation import connected_sets
            csets = connected_sets(transition_matrix)
            if len(csets) > 1:  # disconnected - reduce to largest connected set.
                transition_matrix = transition_matrix[csets[0], :][:, csets[0]]
                observation_probabilities = observation_probabilities[csets[0], :]
                self.nstates = len(csets[0])

        # find observable set
        if self.observe_active:  # cut down observation probabilities to active set
            observation_probabilities = observation_probabilities[:, msm_init.active_set]
            observation_probabilities /= observation_probabilities.sum(axis=1)[:,None]  # renormalize

        # parametrize self
        self._dtrajs_full = dtrajs
        self._dtrajs_lagged = dtrajs_lagged
        self._observable_set = observable_set
        self._dtrajs_obs = dtrajs_obs
        self.set_model_params(P=transition_matrix, pobs=observation_probabilities,
                              reversible=self.reversible, dt_model=self.timestep_traj.get_scaled(self.lag))

        return self

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