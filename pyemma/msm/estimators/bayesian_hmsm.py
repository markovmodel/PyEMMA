__author__ = 'noe'

import numpy as _np

from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _HMSMEstimator
from pyemma.msm.models.hmsm_estimated import EstimatedHMSM as _EstimatedHMSM
from pyemma._base.estimator import Estimator as _Estimator

class BayesianHMSM(_Estimator):
    """Estimator for a Bayesian HMSM

    """
    def __init__(self, lag=1, nstates=2, nsample=1000, init_hmsm=None, reversible=True, connectivity='largest',
                 observe_active=True, dt='1 step', conf=0.683):
        """
        Parameters
        ----------
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        nstates : int, optional, default=2
            number of hidden states
        hmsm : :class:`HMSM <pyemma.msm.ui.hmsm.HMSM>`
            Single-point estimate of HMSM object around which errors will be evaluated
        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.

        """
        self.lag = lag
        self.nstates = nstates
        self.nsample = nsample
        self.init_hmsm = init_hmsm
        self.reversible = reversible
        self.connectivity = connectivity
        self.observe_active = observe_active
        self.dt = dt
        self.conf = conf

    def _estimate(self, dtrajs):
        """

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # if no initial MSM is given, estimate it now
        if self.init_hmsm is None:
            # estimate with store_data=True, because we need an EstimatedHMSM
            hmsm_estimator = _HMSMEstimator(lag=self.lag, nstates=self.nstates, reversible=self.reversible,
                                            connectivity=self.connectivity, observe_active=self.observe_active,
                                            dt=self.dt, store_data=True)
            init_hmsm = hmsm_estimator.estimate(dtrajs)
        else:
            # check input
            assert isinstance(self.init_hmsm, _EstimatedHMSM), 'hmsm must be of type EstimatedHMSM'
            init_hmsm = self.init_hmsm
            self.nstates = init_hmsm.nstates
            self.reversible = init_hmsm.is_reversible

        # if needed, blow up output matrix
        if self.observe_active:
            pobs = _np.zeros((init_hmsm.nstates, init_hmsm.nstates_obs))
            pobs[:, init_hmsm.observable_set] = init_hmsm.observation_probabilities
        else:
            pobs = init_hmsm.observation_probabilities

        # HMM sampler
        from bhmm import discrete_hmm, bayesian_hmm
        hmm_mle = discrete_hmm(init_hmsm.transition_matrix, pobs, stationary=True, reversible=self.reversible)
        print dtrajs
        sampled_hmm = bayesian_hmm(dtrajs, hmm_mle, nsample=self.nsample, transition_matrix_prior='init')

        # Samples
        sample_Ps = [sampled_hmm.sampled_hmms[i].transition_matrix for i in range(self.nsample)]
        sample_mus = [sampled_hmm.sampled_hmms[i].stationary_distribution for i in range(self.nsample)]
        sample_pobs = [sampled_hmm.sampled_hmms[i].output_model.output_probabilities for i in range(self.nsample)]
        for i in range(self.nsample):  # restrict to observable set if necessary
            Bobs = sample_pobs[i][:, init_hmsm.observable_set]
            sample_pobs[i] = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize

        # construct our HMM object
        from pyemma.msm.models.hmsm_sampled import SampledHMSM
        sampled_hmsm = SampledHMSM(init_hmsm, sample_Ps, sample_mus, sample_pobs, conf=self.conf)
        return sampled_hmsm
