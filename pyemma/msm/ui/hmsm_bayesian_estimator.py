__author__ = 'noe'

from pyemma.msm.ui.hmsm_estimator import HMSMEstimator as _HMSMEstimator
from pyemma.msm.ui.hmsm_estimated import EstimatedHMSM as _EstimatedHMSM
from pyemma.util.log import getLogger


class BayesianHMSMEstimator:
    """Estimator for a Bayesian HMSM

    """
    def __init__(self, dtrajs, reversible=True, sparse=False, connectivity='largest',
                 observe_active=True, dt='1 step', conf=0.683, **kwargs):
        """
        Parameters
        ----------
        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.

        """
        # MSM estimation parameters
        self._dtrajs = dtrajs
        self._reversible = reversible
        self._msm_sparse = sparse
        self._msm_connectivity = connectivity
        self._dt = dt
        self._kwargs = kwargs
        # HMM estimation parameters
        self._observe_active = observe_active
        # BHMM estimation parameters
        self._conf = conf
        # other parameters
        self._estimated = False

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self, lag=1, nstates=2, nsample=1000, hmsm=None):
        """

        Parameters
        ----------
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        nstates : int, optional, default=2
            number of hidden states
        hmsm : :class:`HMSM <pyemma.msm.ui.hmsm.HMSM>`
            Single-point estimate of HMSM object around which errors will be evaluated

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # set estimation parameters
        self._lag = lag
        self._nstates = nstates

        # if no initial MSM is given, estimate it now
        if hmsm is None:
            hmsm_estimator = _HMSMEstimator(self._dtrajs, reversible=self._reversible, sparse=self._msm_sparse,
                                            connectivity=self._msm_connectivity, observe_active=self._observe_active,
                                            dt=self._dt, **self._kwargs)
            hmsm = hmsm_estimator.estimate(lag=lag, nstates=nstates)
        else:
            # check input
            assert isinstance(hmsm, _EstimatedHMSM), 'hmsm must be of type EstimatedHMSM'

        # HMM sampler
        from bhmm import discrete_hmm, bayesian_hmm
        hmm_mle = discrete_hmm(hmsm.transition_matrix, hmsm.observation_probabilities,
                               stationary=True, reversible=self._reversible)
        sampled_hmm = bayesian_hmm(self._dtrajs, hmm_mle, nsample=nsample)

        # Samples
        sample_Ps = [sampled_hmm.sampled_hmms[i].transition_matrix for i in range(nsample)]
        sample_mus = [sampled_hmm.sampled_hmms[i].stationary_distribution for i in range(nsample)]
        sample_pobs = [sampled_hmm.sampled_hmms[i].output_model.output_probabilities for i in range(nsample)]
        for i in range(nsample):  # restrict to observable set if necessary
            Bobs = sample_pobs[i][:, hmsm.observable_set]
            sample_pobs[i] = Bobs / Bobs.sum(axis=1)[:, None]  # renormalize

        # construct our HMM object
        from ui.hmsm_sampled import SampledHMSM
        sampled_hmsm = SampledHMSM(hmsm, sample_Ps, sample_mus, sample_pobs, conf=conf)
        return sampled_hmsm
