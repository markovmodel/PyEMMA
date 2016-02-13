
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


r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""

from __future__ import absolute_import

__docformat__ = "restructuredtext en"

import numpy as _np

from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma.util.annotators import alias, aliased

@aliased
class EstimatedHMSM(_HMSM):

    def __init__(self, dtrajs_full, dtrajs_lagged, dt_model, lagtime, nstates_obs, observable_set, dtrajs_obs,
                 transition_matrix, observation_probabilities, pi=None):
        _HMSM.__init__(self, transition_matrix, observation_probabilities, pi=pi, dt_model=dt_model)
        self.lag = lagtime
        self._nstates_obs = nstates_obs
        self._observable_set = observable_set
        self._dtrajs_full = dtrajs_full
        self._dtrajs_lagged = dtrajs_lagged
        self._dtrajs_obs = dtrajs_obs

    @property
    def lagtime(self):
        """ The lag time in steps """
        return self.lag

    @property
    def nstates_obs(self):
        r""" Number of states in discrete trajectories """
        return self._nstates_obs

    @property
    def active_set(self):
        """
        The active set of hidden states on which all hidden state computations are done

        """
        if hasattr(self, '_active_set'):
            return self._active_set
        else:
            return _np.arange(self.nstates)  # all hidden states are active.

    @property
    def observable_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        return self._observable_set

    @property
    @alias('dtrajs_full')
    def discrete_trajectories_full(self):
        """
        A list of integer arrays with the original trajectories.

        """
        return self._dtrajs_full

    @property
    @alias('dtrajs_lagged')
    def discrete_trajectories_lagged(self):
        """
        Transformed original trajectories that are used as an input into the HMM estimation

        """
        return self._dtrajs_lagged

    @property
    @alias('dtrajs_obs')
    def discrete_trajectories_obs(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the observation mode used.
        When using observe_active = True, the indexes will be given on the MSM active set. Frames that are not in the
        observation set will be -1. When observe_active = False, this attribute is identical to
        discrete_trajectories_full

        """
        return self._dtrajs_obs

    ################################################################################
    # Submodel functions using estimation information (counts)
    ################################################################################

    def submodel(self, states=None, obs=None, mincount_connectivity='1/n'):
        """Returns a HMM with restricted state space

        Parameters
        ----------
        states : None, str or int-array
            Hidden states to restrict the model to. In addition to specifying
            the subset, possible options are:
            * None : all states - don't restrict
            * 'populous-strong' : strongly connected subset with maximum counts
            * 'populous-weak' : weakly connected subset with maximum counts
            * 'largest-strong' : strongly connected subset with maximum size
            * 'largest-weak' : weakly connected subset with maximum size
        obs : None, str or int-array
            Observed states to restrict the model to. In addition to specifying
            the subset, possible options are:
            * None : all observed states - don't restrict
            * 'nonempty' : all states with at least one observation
        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. Default value:
            1/nstates.

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if states is None and obs is None and mincount_connectivity==0:
            return self
        if states is None:
            states = _np.arange(self.nstates)
        if obs is None:
            obs = _np.arange(self.nstates_obs)

        if str(mincount_connectivity) == '1/n':
            mincount_connectivity = 1.0/float(self.nstates)

        # handle new connectivity
        from bhmm.estimators import _tmatrix_disconnected
        S = _tmatrix_disconnected.connected_sets(self.count_matrix,
                                                     mincount_connectivity=mincount_connectivity,
                                                     strong=True)
        if len(S) > 1:
            # keep only non-negligible transitions
            C = _np.zeros(self.count_matrix.shape)
            large = _np.where(self.count_matrix >= mincount_connectivity)
            C[large] = self.count_matrix[large]
            for s in S:  # keep all (also small) transition counts within strongly connected subsets
                C[_np.ix_(s, s)] = self.count_matrix[_np.ix_(s, s)]
            # re-estimate transition matrix with disc.
            P = _tmatrix_disconnected.estimate_P(C, reversible=self.reversible, mincount_connectivity=0)
            pi = _tmatrix_disconnected.stationary_distribution(P, C)
        else:
            C = self.count_matrix
            P = self.transition_matrix
            pi = self.stationary_distribution

        # determine substates
        if isinstance(states, str):
            from bhmm.estimators import _tmatrix_disconnected
            strong = 'strong' in states
            largest = 'largest' in states
            S = _tmatrix_disconnected.connected_sets(self.count_matrix, mincount_connectivity=mincount_connectivity,
                                                     strong=strong)
            if largest:
                score = [len(s) for s in S]
            else:
                score = [self.count_matrix[_np.ix_(s, s)].sum() for s in S]
            states = _np.array(S[_np.argmax(score)])
        if states is not None:  # sub-transition matrix
            C = C[_np.ix_(states, states)].copy()
            P = P[_np.ix_(states, states)].copy()
            P /= P.sum(axis=1)[:, None]
            pi = _tmatrix_disconnected.stationary_distribution(P, C)

        # determine observed states
        if str(obs) == 'nonempty':
            import msmtools.estimation as msmest
            obs = _np.where(msmest.count_states(self.discrete_trajectories_full) > 0)[0]
        if obs is not None:
            # full2active mapping
            _full2active = -1 * _np.ones(self.nstates_obs, dtype=int)
            _full2active[obs] = _np.arange(len(obs), dtype=int)
            # observable trajectories
            dtrajs_obs = []
            for dtraj in self.discrete_trajectories_full:
                dtrajs_obs.append(_full2active[dtraj])
            # observation matrix
            B = self.observation_probabilities[_np.ix_(states, obs)].copy()
            B /= B.sum(axis=1)[:, None]
        else:
            dtrajs_obs = self.discrete_trajectories_obs
            B = self.observation_probabilities

        est_hmsm = EstimatedHMSM(self.discrete_trajectories_full, self.discrete_trajectories_lagged,
                                 self.get_model_params()['dt_model'], self.lagtime, _np.size(obs), obs,
                                 dtrajs_obs, P, B, pi=pi)
        est_hmsm._active_set = states
        est_hmsm.count_matrix_EM = self.count_matrix[_np.ix_(states, states)]  # unchanged count matrix
        est_hmsm.count_matrix = C  # count matrix consistent with P
        est_hmsm.initial_count = self.initial_count[states]
        est_hmsm.initial_distribution = self.initial_distribution[states] / self.initial_distribution[states].sum()
        est_hmsm.likelihoods = self.likelihoods  # Likelihood history
        est_hmsm.likelihood = self.likelihood
        est_hmsm.hidden_state_probabilities = self.hidden_state_probabilities  # gamma variables
        est_hmsm.hidden_state_trajectories = self.hidden_state_trajectories  # Viterbi path

        return est_hmsm

    def submodel_largest(self, strong=True, connectivity_mincount='1/n'):
        """ Returns the largest connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if strong:
            return self.submodel(states='largest-strong')
        else:
            return self.submodel(states='largest-weak')

    def submodel_populous(self, strong=True, connectivity_mincount='1/n'):
        """ Returns the most populous connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if strong:
            return self.submodel(states='populous-strong')
        else:
            return self.submodel(states='populous-weak')

    def submodel_disconnect(self, mincount_connectivity='1/n'):
        """Disconnects sets of hidden states that are barely connected

        Runs a connectivity check excluding all transition counts below
        mincount_connectivity. The transition matrix and stationary distribution
        will be re-estimated. Note that the resulting transition matrix
        may have both strongly and weakly connected subsets.

        Parameters
        ----------
        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/nstates.

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        return self.submodel(mincount_connectivity=mincount_connectivity)


    ################################################################################
    # TODO: there is redundancy between this code and EstimatedMSM
    ################################################################################

    def trajectory_weights(self):
        r"""Uses the HMSM to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MSM will be used to reweigh all states.
        Returns a list of weight arrays, one for each trajectory, and with a number of elements equal to
        trajectory frames. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`, this function
        returns corresponding weights:
        .. math::

            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        that are normalized to one:
        .. math::

            \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} = 1

        Suppose you are interested in computing the expectation value of a function :math:`a(x)`, where :math:`x`
        are your input configurations. Use this function to compute the weights of all input configurations and
        obtain the estimated expectation by:
        .. math::

            \langle a \rangle = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t})

        Or if you are interested in computing the time-lagged correlation between functions :math:`a(x)` and
        :math:`b(x)` you could do:
        .. math::

            \langle a(t) b(t+\tau) \rangle_t = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t}) a(x_{i,t+\tau})

        Returns
        -------
        The normalized trajectory weights. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`,
        returns the corresponding weights:
        .. math::

            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        """
        # compute stationary distribution, expanded to full set
        statdist = self.stationary_distribution
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in self.discrete_trajectories_obs:
            w = statdist[dtraj]
            W.append(w)
            wtot += _np.sum(W)
        # normalize
        for w in W:
            w /= wtot
        # done
        return W

    ################################################################################
    # Generation of trajectories and samples
    ################################################################################

    @property
    def observable_state_indexes(self):
        """
        Ensures that the observable states are indexed and returns the indices
        """
        try:  # if we have this attribute, return it
            return self._observable_state_indexes
        except:  # didn't exist? then create it.
            import pyemma.util.discrete_trajectories as dt

            self._observable_state_indexes = dt.index_states(self.discrete_trajectories_obs)
            return self._observable_state_indexes

    # TODO: generate_traj. How should that be defined? Probably indexes of observable states, but should we specify
    #                      hidden or observable states as start and stop states?
    # TODO: sample_by_state. How should that be defined?

    def sample_by_observation_probabilities(self, nsample):
        r"""Generates samples according to given probability distributions

        Parameters
        ----------
        distributions : list or array of ndarray ( (n) )
            m distributions over states. Each distribution must be of length n and must sum up to 1.0
        nsample : int
            Number of samples per distribution. If replace = False, the number of returned samples per state could be
            smaller if less than nsample indexes are available for a state.

        Returns
        -------
        indexes : length m list of ndarray( (nsample, 2) )
            List of the sampled indices by distribution.
            Each element is an index array with a number of rows equal to nsample, with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

        """
        import pyemma.util.discrete_trajectories as dt
        return dt.sample_indexes_by_distribution(self.observable_state_indexes, self.observation_probabilities, nsample)