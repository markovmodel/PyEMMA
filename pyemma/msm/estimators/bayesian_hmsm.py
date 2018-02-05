
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

from __future__ import absolute_import, print_function

import numpy as _np


from pyemma._base.progress import ProgressReporterMixin
from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _MaximumLikelihoodHMSM
from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma.msm.models.hmsm_sampled import SampledHMSM as _SampledHMSM
from pyemma.util.annotators import fix_docs
from pyemma.util.types import ensure_dtraj_list
from pyemma.util.units import TimeUnit
from bhmm import lag_observations as _lag_observations
from msmtools.estimation import number_of_states as _number_of_states

__author__ = 'noe'


@fix_docs
class BayesianHMSM(_MaximumLikelihoodHMSM, _SampledHMSM, ProgressReporterMixin):
    r"""Estimator for a Bayesian Hidden Markov state model"""
    __serialize_version = 0
    __serialize_fields = ('accuracy',
                         'count_matrix',
                         'hidden_state_probabilities',
                         'hidden_state_trajectories',
                         'initial_count',
                         'initial_distribution',
                         'likelihood',
                         'likelihoods',
                         'sampled_trajs',
                         )

    def __init__(self, nstates=2, lag=1, stride='effective',
                 p0_prior='mixed', transition_matrix_prior='mixed',
                 nsamples=100, init_hmsm=None, reversible=True, stationary=False,
                 connectivity='largest', mincount_connectivity='1/n', separate=None, observe_nonempty=True,
                 dt_traj='1 step', conf=0.95, store_hidden=False, show_progress=True):
        r"""Estimator for a Bayesian HMSM

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
                s[0], s[tau], s[2 tau], ...
                s[stride], s[stride + tau], s[stride + 2 tau], ...
            Setting stride = 1 will result in using all data (useful for
            maximum likelihood estimator), while a Bayesian estimator requires
            a longer stride in order to have statistically uncorrelated
            trajectories. Setting stride = None 'effective' uses the largest
            neglected timescale as an estimate for the correlation time and
            sets the stride accordingly.
        p0_prior : None, str, float or ndarray(n)
            Prior for the initial distribution of the HMM. Will only be active
            if stationary=False (stationary=True means that p0 is identical to
            the stationary distribution of the transition matrix).
            Currently implements different versions of the Dirichlet prior that
            is conjugate to the Dirichlet distribution of p0. p0 is sampled from:

            .. math:
                p0 \sim \prod_i (p0)_i^{a_i + n_i - 1}
            where :math:`n_i` are the number of times a hidden trajectory was in
            state :math:`i` at time step 0 and :math:`a_i` is the prior count.
            Following options are available:

            * 'mixed' (default),  :math:`a_i = p_{0,init}`, where :math:`p_{0,init}`
              is the initial distribution of initial_model.
            *  ndarray(n) or float,
               the given array will be used as A.
            *  'uniform',  :math:`a_i = 1`
            *   None,  :math:`a_i = 0`. This option ensures coincidence between
                sample mean an MLE. Will sooner or later lead to sampling problems,
                because as soon as zero trajectories are drawn from a given state,
                the sampler cannot recover and that state will never serve as a starting
                state subsequently. Only recommended in the large data regime and
                when the probability to sample zero trajectories from any state
                is negligible.
        transition_matrix_prior : str or ndarray(n, n)
            Prior for the HMM transition matrix.
            Currently implements Dirichlet priors if reversible=False and reversible
            transition matrix priors as described in [3]_ if reversible=True. For the
            nonreversible case the posterior of transition matrix :math:`P` is:

            .. math:
                P \sim \prod_{i,j} p_{ij}^{b_{ij} + c_{ij} - 1}
            where :math:`c_{ij}` are the number of transitions found for hidden
            trajectories and :math:`b_{ij}` are prior counts.

            * 'mixed' (default),  :math:`b_{ij} = p_{ij,init}`, where :math:`p_{ij,init}`
              is the transition matrix of initial_model. That means one prior
              count will be used per row.
            * ndarray(n, n) or broadcastable,
              the given array will be used as B.
            * 'uniform',  :math:`b_{ij} = 1`
            * None,  :math:`b_ij = 0`. This option ensures coincidence between
              sample mean an MLE. Will sooner or later lead to sampling problems,
              because as soon as a transition :math:`ij` will not occur in a
              sample, the sampler cannot recover and that transition will never
              be sampled again. This option is not recommended unless you have
              a small HMM and a lot of data.
        init_hmsm : :class:`HMSM <pyemma.msm.models.HMSM>`, default=None
            Single-point estimate of HMSM object around which errors will be evaluated.
            If None is give an initial estimate will be automatically generated using the
            given parameters.
        store_hidden : bool, optional, default=False
            store hidden trajectories in sampled HMMs
        show_progress : bool, default=True
            Show progressbars for calculation?

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of complex
            molecules. J. Chem. Phys. 139, 184114 (2013)
        .. [2] J. D. Chodera Et Al: Bayesian hidden Markov model analysis of
            single-molecule force spectroscopy: Characterizing kinetics under
            measurement uncertainty. arXiv:1108.1430 (2011)
        .. [3] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe:
            Estimation and uncertainty of reversible Markov models.
            J. Chem. Phys. 143, 174101 (2015).

        """
        super(BayesianHMSM, self).__init__(nstates=nstates, lag=lag, stride=stride,
                                           reversible=reversible, stationary=stationary,
                                           connectivity=connectivity, mincount_connectivity=mincount_connectivity,
                                           observe_nonempty=observe_nonempty, separate=separate,
                                           dt_traj=dt_traj)
        self.p0_prior = p0_prior
        self.transition_matrix_prior = transition_matrix_prior
        self.nsamples = nsamples
        if init_hmsm is not None:
            assert issubclass(init_hmsm.__class__, _MaximumLikelihoodHMSM), 'hmsm must be of type MaximumLikelihoodHMSM'
        self.init_hmsm = init_hmsm
        self.conf = conf
        self.store_hidden = store_hidden
        self.show_progress = show_progress

    def _estimate(self, dtrajs):
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)

        if self.init_hmsm is None:  # estimate using maximum-likelihood superclass
            # memorize the observation state for bhmm and reset
            # TODO: more elegant solution is to set Estimator params only temporarily in estimate(X, **kwargs)
            default_connectivity = self.connectivity
            default_mincount_connectivity = self.mincount_connectivity
            default_observe_nonempty = self.observe_nonempty
            self.connectivity = None
            self.observe_nonempty = False
            self.mincount_connectivity = 0
            self.accuracy = 1e-2  # this is sufficient for an initial guess
            super(BayesianHMSM, self)._estimate(dtrajs)
            self.connectivity = default_connectivity
            self.mincount_connectivity = default_mincount_connectivity
            self.observe_nonempty = default_observe_nonempty
        else:  # if given another initialization, must copy its attributes
            copy_attributes = ['_nstates', '_reversible', '_pi', '_observable_set', 'likelihoods', 'likelihood',
                               'hidden_state_probabilities', 'hidden_state_trajectories', 'count_matrix',
                               'initial_count', 'initial_distribution', '_active_set']
            check_user_choices = ['lag', '_nstates']

            # check if nstates and lag are compatible
            for attr in check_user_choices:
                if not self.__getattribute__(attr) == self.init_hmsm.__getattribute__(attr):
                    raise UserWarning('BayesianHMSM cannot be initialized with init_hmsm with '
                                      + 'incompatible lag or nstates.')

            if not _np.array_equal(dtrajs, self.init_hmsm._dtrajs_full):
                raise NotImplementedError('Bayesian HMM estimation with init_hmsm is currently only implemented ' +
                                          'if applied to the same data.')

            # TODO: implement more elegant solution to copy-pasting effective stride evaluation from ML HMM.
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
                    corrtime = max(1, msm_nr.timescales()[self.nstates - 1])
                    # use the smaller of these two pessimistic estimates
                    self.stride = int(min(self.lag, 2 * corrtime))

            # if stride is different to init_hmsm, check if microstates in lagged-strided trajs are compatible
            if self.stride != self.init_hmsm.stride:
                dtrajs_lagged_strided = _lag_observations(dtrajs, self.lag, stride=self.stride)
                _nstates_obs = _number_of_states(dtrajs_lagged_strided, only_used=True)
                _nstates_obs_full = _number_of_states(dtrajs)

                if _np.setxor1d(_np.concatenate(dtrajs_lagged_strided),
                                 _np.concatenate(self.init_hmsm._dtrajs_lagged)).size != 0:
                    raise UserWarning('Choice of stride has excluded a different set of microstates than in ' +
                                      'init_hmsm. Set of observed microstates in time-lagged strided trajectories ' +
                                      'must match to the one used for init_hmsm estimation.')

                self._dtrajs_full = dtrajs
                self._dtrajs_lagged = dtrajs_lagged_strided
                self._nstates_obs_full = _nstates_obs_full
                self._nstates_obs = _nstates_obs
                self._observable_set = _np.arange(self._nstates_obs)
                self._dtrajs_obs = dtrajs
            else:
                copy_attributes += ['_dtrajs_full', '_dtrajs_lagged', '_nstates_obs_full',
                                    '_nstates_obs', '_observable_set', '_dtrajs_obs']

            # update self with estimates from init_hmsm
            self.__dict__.update(
                {k: i for k, i in self.init_hmsm.__dict__.items() if k in copy_attributes})

            # as mentioned in the docstring, take init_hmsm observed set observation probabilities
            self.observe_nonempty = False

            # update HMM Model
            self.update_model_params(P=self.init_hmsm.transition_matrix, pobs=self.init_hmsm.observation_probabilities,
                                     dt_model=TimeUnit(self.dt_traj).get_scaled(self.lag))

        # check if we have a valid initial model
        import msmtools.estimation as msmest
        if self.reversible and not msmest.is_connected(self.count_matrix):
            raise NotImplementedError('Encountered disconnected count matrix:\n ' + str(self.count_matrix)
                                      + 'with reversible Bayesian HMM sampler using lag=' + str(self.lag)
                                      + ' and stride=' + str(self.stride) + '. Consider using shorter lag, '
                                      + 'or shorter stride (to use more of the data), '
                                      + 'or using a lower value for mincount_connectivity.')

        # here we blow up the output matrix (if needed) to the FULL state space because we want to use dtrajs in the
        # Bayesian HMM sampler. This is just an initialization.
        nstates_full = msmest.number_of_states(dtrajs)
        if self.nstates_obs < nstates_full:
            eps = 0.01 / nstates_full  # default output probability, in order to avoid zero columns
            # full state space output matrix. make sure there are no zero columns
            B_init = eps * _np.ones((self.nstates, nstates_full), dtype=_np.float64)
            # fill active states
            B_init[:, self.observable_set] = _np.maximum(eps, self.observation_probabilities)
            # renormalize B to make it row-stochastic
            B_init /= B_init.sum(axis=1)[:, None]
        else:
            B_init = self.observation_probabilities

        # HMM sampler
        if self.show_progress:
            self._progress_register(self.nsamples, description='Sampling HMSMs', stage=0)

            def call_back():
                self._progress_update(1, stage=0)
        else:
            call_back = None

        from bhmm import discrete_hmm, bayesian_hmm

        if self.init_hmsm is not None:
            hmm_mle = self.init_hmsm.hmm
        else:
            hmm_mle = discrete_hmm(self.initial_distribution, self.transition_matrix, B_init)

        sampled_hmm = bayesian_hmm(self.discrete_trajectories_lagged, hmm_mle, nsample=self.nsamples,
                                   reversible=self.reversible, stationary=self.stationary,
                                   p0_prior=self.p0_prior, transition_matrix_prior=self.transition_matrix_prior,
                                   store_hidden=self.store_hidden, call_back=call_back)

        if self.show_progress:
            self._progress_force_finish(stage=0)

        # Samples
        sample_inp = [(m.transition_matrix, m.stationary_distribution, m.output_probabilities)
                      for m in sampled_hmm.sampled_hmms]

        samples = []
        for P, pi, pobs in sample_inp:  # restrict to observable set if necessary
            Bobs = pobs[:, self.observable_set]
            pobs = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize
            samples.append(_HMSM(P, pobs, pi=pi, dt_model=self.dt_model))

        # store results
        self.sampled_trajs = [sampled_hmm.sampled_hmms[i].hidden_state_trajectories for i in range(self.nsamples)]
        self.update_model_params(samples=samples)

        # deal with connectivity
        states_subset = None
        if self.connectivity == 'largest':
            states_subset = 'largest-strong'
        elif self.connectivity == 'populous':
            states_subset = 'populous-strong'
        # OBSERVATION SET
        if self.observe_nonempty:
            observe_subset = 'nonempty'
        else:
            observe_subset = None

        # return submodel (will return self if all None)
        return self.submodel(states=states_subset, obs=observe_subset,
                             mincount_connectivity=self.mincount_connectivity)

    def submodel(self, states=None, obs=None, mincount_connectivity='1/n'):
        # call submodel on MaximumLikelihoodHMSM
        _MaximumLikelihoodHMSM.submodel(self, states=states, obs=obs, mincount_connectivity=mincount_connectivity)
        # if samples set, also reduce them
        if hasattr(self, 'samples') and self.samples is not None:
            subsamples = [sample.submodel(states=self.active_set, obs=self.observable_set)
                          for sample in self.samples]
            self.update_model_params(samples=subsamples)

        # return
        return self
