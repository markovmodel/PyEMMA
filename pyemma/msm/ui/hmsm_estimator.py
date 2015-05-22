__author__ = 'noe'

import numpy as np

from pyemma.msm.ui.msm_estimated import EstimatedMSM as _EstimatedMSM
from pyemma.msm.ui.msm_estimator import MSMEstimator as _MSMEstimator
from pyemma.msm.ui.hmsm_estimated import EstimatedHMSM as _EstimatedHMSM
from pyemma.util.log import getLogger
from pyemma.util import types as _types


class HMSMEstimator:
    """ML Estimator for a HMSM given a MSM

    """
    def __init__(self, dtrajs, reversible=True, sparse=False, connectivity='largest',
                 observe_active=True, dt='1 step', **kwargs):
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
        # other parameters
        self._estimated = False

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self, lag=1, nstates=2, msm_init=None):
        """

        Parameters
        ----------
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        nstates : int, optional, default=2
            number of hidden states
        msm_init : :class:`MSM <pyemma.msm.ui.msm_estimated.MSM>`
            MSM object to initialize the estimation

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # set estimation parameters
        self._lag = lag
        self._nstates = nstates

        # if no initial MSM is given, estimate it now
        if msm_init is None:
            msm_estimator = _MSMEstimator(self._dtrajs, reversible=self._reversible, sparse=self._msm_sparse,
                                          connectivity=self._msm_connectivity, dt=self._dt, **self._kwargs)
            msm_init = msm_estimator.estimate(lag=lag)
        else:
            assert isinstance(msm_init, _EstimatedMSM), 'msm_init must be of type EstimatedMSM'

        # check input
        assert _types.is_int(nstates) and nstates > 1 and nstates <= msm_init.nstates, \
            'nstates must be an int in [2,msmobj.nstates]'
        timescale_ratios = msm_init.timescales()[:-1] / msm_init.timescales()[1:]
        if timescale_ratios[nstates-2] < 2.0:
            self._logger.warn('Requested coarse-grained model with ' + str(nstates) + ' metastable states. ' +
                              'The ratio of relaxation timescales between ' + str(nstates) + ' and ' + str(nstates+1) +
                              ' states is only ' + str(timescale_ratios[nstates-2]) + ' while we recomment at ' +
                              'least 2. It is possible that the resulting HMM is inaccurate. Handle with caution.')

        # set things from MSM
        self._nstates_obs_full = msm_init.nstates_full
        if self._observe_active:
            nstates_obs = msm_init.nstates
            observable_set = msm_init.active_set
            dtrajs_obs = msm_init.discrete_trajectories_active
        else:
            nstates_obs = msm_init.nstates_full
            observable_set = np.arange(self._nstates_obs_full)
            dtrajs_obs = msm_init.discrete_trajectories_full

        # TODO: this is redundant with BHMM code because that code is currently not easily accessible and
        # TODO: we don't want to re-estimate. Should be reengineered in bhmm.
        # ---------------------------------------------------------------------------------------
        # PCCA-based coarse-graining
        # ---------------------------------------------------------------------------------------
        # pcca- to number of metastable states
        pcca = msm_init.pcca(self._nstates)

        # HMM output matrix
        B_conn = msm_init.metastable_distributions
        # full state space output matrix
        eps = 0.01 * (1.0/self._nstates_obs_full)  # default output probability, in order to avoid zero columns
        B = eps * np.ones((self._nstates, self._nstates_obs_full), dtype=np.float64)
        # expand B_conn to full state space
        B[:, msm_init.active_set] = B_conn[:, :]
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:, None]

        # coarse-grained transition matrix
        P_coarse = pcca.coarse_grained_transition_matrix
        # take care of unphysical values. First symmetrize
        X = np.dot(np.diag(pcca.coarse_grained_stationary_probability), P_coarse)
        X = 0.5*(X + X.T)
        # if there are values < 0, set to eps
        X = np.maximum(X, eps)
        # turn into coarse-grained transition matrix
        A = X / X.sum(axis=1)[:, None]

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------
        # lazy import bhmm here in order to avoid dependency loops
        import bhmm
        # initialize discrete HMM
        self.hmm_init = bhmm.discrete_hmm(A, B, stationary=True, reversible=self._reversible)
        # run EM
        hmm = bhmm.estimate_hmm(msm_init.discrete_trajectories_full, self._nstates,
                                lag=msm_init.lagtime, initial_model=self.hmm_init)
        self.hmm = bhmm.DiscreteHMM(hmm)

        # find observable set
        transition_matrix = self.hmm.transition_matrix
        observation_probabilities = self.hmm.output_probabilities
        if self._observe_active:  # cut down observation probabilities to active set
            observation_probabilities = observation_probabilities[:, msm_init.active_set]
            observation_probabilities /= observation_probabilities.sum(axis=1)[:,None]  # renormalize

        # construct result
        self._hmsm = _EstimatedHMSM(self._dtrajs, self._dt, lag,
                                    nstates_obs, observable_set, dtrajs_obs, transition_matrix,
                                    observation_probabilities)
        return self._hmsm

