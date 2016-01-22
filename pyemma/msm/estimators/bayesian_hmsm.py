
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
from six.moves import range

import numpy as _np
from pyemma.util.types import ensure_dtraj_list
from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _MaximumLikelihoodHMSM
from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma.msm.estimators.estimated_hmsm import EstimatedHMSM as _EstimatedHMSM
from pyemma.msm.models.hmsm_sampled import SampledHMSM as _SampledHMSM
from pyemma.util.units import TimeUnit
from pyemma._base.progress import ProgressReporter

__author__ = 'noe'


class BayesianHMSM(_MaximumLikelihoodHMSM, _SampledHMSM, ProgressReporter):
    r"""Estimator for a Bayesian Hidden Markov state model"""

    def __init__(self, nstates=2, lag=1, stride='effective', prior='mixed',
                 nsamples=100, init_hmsm=None, reversible=True,
                 connectivity='largest', observe_active=True,
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
        prior : str, optional, default='mixed'
            prior used in the estimation of the transition matrix. While 'sparse'
            would be preferred as it doesn't bias the distribution way from the
            maximum-likelihood, this prior is sensitive to loss of connectivity.
            Loss of connectivity can occur in the Gibbs sampling algorithm used
            here because in each iteration the hidden state sequence is randomly
            generated. Once full connectivity is lost in one of these steps, the
            current algorithm cannot recover from that. As a solution we suggest
            using a prior that ensures that the estimated transition matrix is
            connected even if the sampled state sequence is not.

            * 'sparse' : the sparse prior proposed in [1]_ which centers the
              posterior around the maximum likelihood estimator. This is the
              preferred option if there are no connectivity problems. However
              this prior is sensitive to loss of connectivity.
            * 'uniform' : uniform prior probability for every transition matrix
              element. Compared to the sparse prior, 'uniform' adds +1 to
              every transition count. Weak prior that ensures connectivity,
              but can lead to large biases if some states have small exit
              probabilities.
            * 'mixed' : ensures connectivity by adding a prior taken from the
              maximum likelihood estimate (MLE) of the hidden transition
              matrix P. The rows of P are scaled in order to have total
              outgoing transition counts of at least 1 out of each state.
              While this operation centers the posterior around the MLE, it
              can be a very strong prior if states with small exit
              probabilities are involved, and can therefore artificially
              reduce the error bars.
        init_hmsm : :class:`HMSM <pyemma.msm.models.HMSM>`, default=None
            Single-point estimate of HMSM object around which errors will be evaluated.
            If None is give an initial estimate will be automatically generated using the
            given parameters.
        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.
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

        """
        self.lag = lag
        self.stride = stride
        self.nstates = nstates
        self.prior = prior
        self.nsamples = nsamples
        self.init_hmsm = init_hmsm
        self.reversible = reversible
        self.connectivity = connectivity
        self.observe_active = observe_active
        self.dt_traj = dt_traj
        self.timestep_traj = TimeUnit(dt_traj)
        self.conf = conf
        self.store_hidden = store_hidden
        self.show_progress = show_progress

    def _estimate(self, dtrajs):
        """

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.estimators.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)

        # if no initial MSM is given, estimate it now
        if self.init_hmsm is None:
            # estimate with store_data=True, because we need an EstimatedHMSM
            hmsm_estimator = _MaximumLikelihoodHMSM(lag=self.lag, stride=self.stride, nstates=self.nstates,
                                            reversible=self.reversible, connectivity=self.connectivity,
                                            observe_active=self.observe_active, dt_traj=self.dt_traj)
            init_hmsm = hmsm_estimator.estimate(dtrajs)  # estimate with lagged trajectories
            self.nstates = init_hmsm.nstates  # might have changed due to connectivity
        else:
            # check input
            assert isinstance(self.init_hmsm, _EstimatedHMSM), 'hmsm must be of type EstimatedHMSM'
            init_hmsm = self.init_hmsm
            self.nstates = init_hmsm.nstates
            self.reversible = init_hmsm.is_reversible

        # here we blow up the output matrix (if needed) to the FULL state space because we want to use dtrajs in the
        # Bayesian HMM sampler
        if self.observe_active:
            import msmtools.estimation as msmest
            nstates_full = msmest.number_of_states(dtrajs)
            # pobs = _np.zeros((init_hmsm.nstates, nstates_full))  # currently unused because that produces zero cols
            eps = 0.01 / nstates_full  # default output probability, in order to avoid zero columns
            # full state space output matrix. make sure there are no zero columns
            pobs = eps * _np.ones((self.nstates, nstates_full), dtype=_np.float64)
            # fill active states
            pobs[:, init_hmsm.observable_set] = _np.maximum(eps, init_hmsm.observation_probabilities)
            # renormalize B to make it row-stochastic
            pobs /= pobs.sum(axis=1)[:, None]
        else:
            pobs = init_hmsm.observation_probabilities

        # HMM sampler
        if self.show_progress:
            self._progress_register(self.nsamples, description='Sampling HMSMs', stage=0)

            def call_back():
                self._progress_update(1, stage=0)
        else:
            call_back = None

        from bhmm import discrete_hmm, bayesian_hmm
        hmm_mle = discrete_hmm(init_hmsm.transition_matrix, pobs, stationary=True, reversible=self.reversible)

        # define prior
        if self.prior == 'sparse':
            self.prior_count_matrix = _np.zeros((self.nstates, self.nstates), dtype=_np.float64)
        elif self.prior == 'uniform':
            self.prior_count_matrix = _np.ones((self.nstates, self.nstates), dtype=_np.float64)
        elif self.prior == 'mixed':
            # C0 = _np.dot(_np.diag(init_hmsm.stationary_distribution), init_hmsm.transition_matrix)
            P0 = init_hmsm.transition_matrix
            P0_offdiag = P0 - _np.diag(_np.diag(P0))
            scaling_factor = 1.0 / _np.sum(P0_offdiag, axis=1)
            self.prior_count_matrix = P0 * scaling_factor[:, None]
        else:
            raise ValueError('Unknown prior mode: '+self.prior)

        sampled_hmm = bayesian_hmm(init_hmsm.discrete_trajectories_lagged, hmm_mle, nsample=self.nsamples,
                                   transition_matrix_prior=self.prior_count_matrix, store_hidden=self.store_hidden,
                                   call_back=call_back)

        if self.show_progress:
            self._progress_force_finish(stage=0)

        # Samples
        sample_Ps = [sampled_hmm.sampled_hmms[i].transition_matrix for i in range(self.nsamples)]
        sample_pis = [sampled_hmm.sampled_hmms[i].stationary_distribution for i in range(self.nsamples)]
        sample_pobs = [sampled_hmm.sampled_hmms[i].output_model.output_probabilities for i in range(self.nsamples)]
        samples = []
        for i in range(self.nsamples):  # restrict to observable set if necessary
            Bobs = sample_pobs[i][:, init_hmsm.observable_set]
            sample_pobs[i] = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize
            samples.append(_HMSM(sample_Ps[i], sample_pobs[i], pi=sample_pis[i], dt_model=init_hmsm.dt_model))

        # store hidden trajectories
        self.sampled_trajs = [sampled_hmm.sampled_hmms[i].hidden_state_trajectories for i in range(self.nsamples)]

        # parametrize self
        self._dtrajs_full = dtrajs
        self._observable_set = init_hmsm._observable_set
        self._dtrajs_obs = init_hmsm._dtrajs_obs
        self.set_model_params(samples=samples, P=init_hmsm.transition_matrix, pobs=init_hmsm.observation_probabilities,
                              dt_model=init_hmsm.dt_model)

        return self