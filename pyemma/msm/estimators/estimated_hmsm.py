
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
                 transition_matrix, observation_probabilities):
        _HMSM.__init__(self, transition_matrix, observation_probabilities, dt_model=dt_model)
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
        return _np.arange(self.nstates)  # currently assume all hidden states are active.

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