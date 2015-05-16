__author__ = 'noe'

import numpy as np

from pyemma.msm import estimation as msmest
from pyemma.msm.ui.msm_estimated import EstimatedMSM as _EstimatedMSM
from pyemma.util.log import getLogger
from pyemma.util import types as _types

class HMSMEstimator:
    """ML Estimator for a HMSM given a MSM

    """
    def __init__(self, msmobj, nstates, observe_active=True, estimate=True):
        """
        Parameters
        ----------
        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.

        """
        # check input
        assert isinstance(msmobj, _EstimatedMSM), 'msmobj must be ob type EstimatedMSM'
        assert _types.is_int(nstates) and nstates > 1 and nstates <= msmobj.nstates, 'nstates must be an int in [2,msmobj.nstates]'

        # set basic parameters
        self._msmobj = msmobj
        self._nstates = nstates
        # restrict observation states to MSM active set
        self._observe_active = observe_active
        # currently only support largest
        # TODO: actually we don't check for connectivity at all so far. This needs to be fixed
        self._connectivity='largest'

        # run estimation
        self._estimated = False
        if estimate:
            self.estimate()

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self):
        # TODO: this is redundant with BHMM code because that code is currently not easily accessible and
        # TODO: we don't want to re-estimate. Should be reengineered in bhmm.
        # ---------------------------------------------------------------------------------------
        # PCCA-based coarse-graining
        # ---------------------------------------------------------------------------------------
        # pcca- to number of metastable states
        pcca = self._msmobj.pcca(self._nstates)

        # HMM output matrix
        B_conn = self._msmobj.metastable_distributions
        # full state space output matrix
        self._nstates_obs_full = self._msmobj.nstates_full
        eps = 0.01 * (1.0/self._nstates_obs_full) # default output probability, in order to avoid zero columns
        B = eps * np.ones((self._nstates, self._nstates_obs_full), dtype=np.float64)
        # expand B_conn to full state space
        B[:,self._msmobj.active_set] = B_conn[:,:]
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:,None]

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
        self.hmm_init = bhmm.discrete_hmm(P_coarse, B, stationary=True, reversible=self._msmobj.is_reversible)
        # run EM
        hmm = bhmm.estimate_hmm(self._msmobj.discrete_trajectories_full, self._nstates,
                                lag=self._msmobj.lagtime, initial_model=self.hmm_init)
        self.hmm = bhmm.DiscreteHMM(hmm)
        # done
        self._estimated = True

    def _assert_estimated(self):
        assert self._estimated, "MSM hasn't been estimated yet, make sure to call estimate()"

    @property
    def estimated(self):
        """Returns whether this msm has been estimated yet"""
        return self._estimated

    @property
    def nstates(self):
        """
        The number of all hidden states

        """
        return self._nstates

    @property
    def nstates_obs(self):
        """
        The number of all hidden states

        """
        if self._observe_active:
            return self._msmobj.nstates
        else:
            return self._nstates_obs_full

    @property
    def observable_set(self):
        """
        Set of observed states

        """
        if self._observe_active:
            return self._msmobj.active_set
        else:
            return np.arange(self._nstates_obs_full)

    @property
    def lagtime(self):
        """ The lag time in steps """
        return self.hmm.lag

    @property
    def is_reversible(self):
        """Returns whether the HMSM is reversible """
        return self.hmm.is_reversible

    @property
    def dt(self):
        """Returns the time step"""
        return self._msmobj.timestep

    @property
    def discrete_trajectories_full(self):
        """
        A list of integer arrays with the original trajectories.

        """
        return self._msmobj.discrete_trajectories_full

    @property
    def discrete_trajectories_obs(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the observation mode used.
        When using observe_active = True, the indexes will be given on the MSM active set. Frames that are not in the
        observation set will be -1. When observe_active = False, this attribute is identical to
        discrete_trajectories_full

        """
        if self._observe_active:
            return self._msmobj.discrete_trajectories_active
        else:
            return self._msmobj.discrete_trajectories_full

    @property
    def transition_matrix(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_estimated()
        return self.hmm.transition_matrix

    @property
    def observation_probabilities(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_estimated()
        if self._observe_active:
            Bfull = self.hmm.output_probabilities
            Bobs = Bfull[:,self._msmobj.active_set]  # restrict to active set
            Bobs /= Bobs.sum(axis=1)[:,None]  # renormalize
            return Bobs
        else:
            return self.hmm.output_probabilities

    @property
    def dt(self):
        return self._msmobj.timestep