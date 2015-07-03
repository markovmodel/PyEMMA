__author__ = 'noe'

import numpy as _np

from pyemma.util.types import ensure_dtraj_list
from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _HMSMEstimator
from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma.msm.estimators.estimated_hmsm import EstimatedHMSM as _EstimatedHMSM
from pyemma.msm.models.hmsm_sampled import SampledHMSM as _SampledHMSM
from pyemma._base.estimator import Estimator as _Estimator

class BayesianHMSM(_HMSMEstimator, _SampledHMSM):
    """Estimator for a Bayesian HMSM

    """
    def __init__(self, nstates=2, lag=1, nsamples=100, init_hmsm=None, reversible=True, connectivity='largest',
                 observe_active=True, dt_traj='1 step', conf=0.95):
        """
        Parameters
        ----------
        nstates : int, optional, default=2
            number of hidden states

        lag : int, optional, default=1
            lagtime to estimate the HMSM at

        hmsm : :class:`HMSM <pyemma.msm.ui.hmsm.HMSM>`
            Single-point estimate of HMSM object around which errors will be evaluated

        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.

        """
        self.lag = lag
        self.nstates = nstates
        self.nsamples = nsamples
        self.init_hmsm = init_hmsm
        self.reversible = reversible
        self.connectivity = connectivity
        self.observe_active = observe_active
        self.dt_traj = dt_traj
        self.conf = conf

    def _estimate(self, dtrajs):
        """

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # if no initial MSM is given, estimate it now
        if self.init_hmsm is None:
            # estimate with store_data=True, because we need an EstimatedHMSM
            hmsm_estimator = _HMSMEstimator(lag=self.lag, nstates=self.nstates, reversible=self.reversible,
                                            connectivity=self.connectivity, observe_active=self.observe_active,
                                            dt_traj=self.dt_traj)
            init_hmsm = hmsm_estimator.estimate(dtrajs)
        else:
            # check input
            assert isinstance(self.init_hmsm, _EstimatedHMSM), 'hmsm must be of type EstimatedHMSM'
            init_hmsm = self.init_hmsm
            self.nstates = init_hmsm.nstates
            self.reversible = init_hmsm.is_reversible

        # here we blow up the output matrix (if needed) to the FULL state space because we want to use dtrajs in the
        # Bayesian HMM sampler
        if self.observe_active:
            import pyemma.msm.estimation as msmest
            nstates_full = msmest.number_of_states(dtrajs)
            pobs = _np.zeros((init_hmsm.nstates, nstates_full))
            pobs[:, init_hmsm.observable_set] = init_hmsm.observation_probabilities
        else:
            pobs = init_hmsm.observation_probabilities

        # HMM sampler
        from bhmm import discrete_hmm, bayesian_hmm
        hmm_mle = discrete_hmm(init_hmsm.transition_matrix, pobs, stationary=True, reversible=self.reversible)
        sampled_hmm = bayesian_hmm(dtrajs, hmm_mle, nsample=self.nsamples, transition_matrix_prior='init-connect')

        # Samples
        sample_Ps = [sampled_hmm.sampled_hmms[i].transition_matrix for i in range(self.nsamples)]
        sample_pis = [sampled_hmm.sampled_hmms[i].stationary_distribution for i in range(self.nsamples)]
        sample_pobs = [sampled_hmm.sampled_hmms[i].output_model.output_probabilities for i in range(self.nsamples)]
        samples = []
        for i in range(self.nsamples):  # restrict to observable set if necessary
            Bobs = sample_pobs[i][:, init_hmsm.observable_set]
            sample_pobs[i] = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize
            samples.append(_HMSM(sample_Ps[i], sample_pobs[i], pi=sample_pis[i], dt_model=init_hmsm.dt_model))

        # parametrize self
        self._dtrajs_full = dtrajs
        self._observable_set = init_hmsm._observable_set
        self._dtrajs_obs = init_hmsm._dtrajs_obs
        self.set_model_params(samples=samples, P=init_hmsm.transition_matrix, pobs=init_hmsm.observation_probabilities)

        return self