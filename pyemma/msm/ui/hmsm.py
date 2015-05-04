
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

__docformat__ = "restructuredtext en"

import numpy as np
import copy
from math import ceil
from bhmm import DiscreteHMM

class HMSM(object):
    r""" Hidden Markov model on discrete states.

    Parameters
    ----------
    hmm : :class:`DiscreteHMM <bhmm.DiscreteHMM>`
        Hidden Markov Model

    """

    # TODO: Do we want the exact same interface like for an MSM? Then we should copy all MSM functions, even
    # TODO: if they are not completely meaningful, such as is_sparse.

    def __init__(self, hmm, dt='1 step'):
        # save underlying HMM
        self._hmm = copy.deepcopy(hmm)
        # set time step
        from pyemma.util.units import TimeUnit
        self._timeunit = TimeUnit(dt)

    @property
    def is_reversible(self):
        """Returns whether the HMSM is reversible """
        return self._hmm.is_reversible

    @property
    def is_sparse(self):
        """Returns False, because a HMSM is not represented by sparse matrices, although it is low-rank """
        return False

    @property
    def timestep(self):
        """Returns the physical time corresponding to one step of the transition matrix as string, e.g. '10 ps'"""
        return str(self._timeunit)

    @property
    def lag(self):
        return self._hmm.lag

    @property
    def nstates(self):
        # TODO: This is ambiguous. We could instead use nstates_hidden and nstates_obs or similar.
        """The number of observable states

        """
        return self._hmm.nsymbols

    @property
    def transition_matrix(self):
        # TODO: This is ambiguous. We could instead use transition_matrix_hidden and transition_matrix_obs or similar.
        # TODO: Moreover, since an HMM is not Markovian in the observed state space it would be better to call
        # TODO: The observed transition matrix with a lag.
        #
        # TODO: Alternative: convention that this is the hidden transition matrix and add the observable transition
        # TODO: matrix in a separate function only defined for HMSMs. Downside is that then transition_matrix really
        # TODO: means different things for MSM and HMSM.
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        raise NotImplementedError('What do we do here?')

    @property
    def metastable_transition_matrix(self):
        # TODO: Or hidden_transition_matrix? Or transition_matrix_hidden?
        return self._hmm.transition_matrix

    def obs_transition_matrix(self, k=1):
        """ Computes the transition matrix between observed states

        Transition matrices for longer lag times than the one used to parametrize this HMSM can be obtained by
        setting the k option. Note that a HMSM is not Markovian, thus we cannot compute transition matrices at longer
        lag times using the Chapman-Kolmogorow equality. I.e.:

        .. math::
            P (k \tau) \neq P^k (\tau)

        This function computes the correct transition matrix using the metastable (coarse) transition matrix
        :math:`P_c` as:

        .. math::
            P (k \tau) = {\Pi}^-1 \chi^{\top} ({\Pi}_c) P_c^k (\tau) \chi

        where :math:`\chi` is the output probability matrix, :math:`\Pi_c` is a diagonal matrix with the
        metastable-state (coarse) stationary distribution and :math:`\Pi` is a diagonal matrix with the
        observable-state stationary distribution.

        Parameters
        ----------
        k : int, optional, default=1
            Multiple of the lag time for which the
            By default (k=1), the transition matrix at the lag time used to construct this HMSM will be returned.
            If a higher power is given,

        """
        Pi_c = np.diag(self._hmm.stationary_distribution)
        P_c = self._hmm.transition_matrix
        P_c_k = np.linalg.matrix_power(P_c, k) # take a power if needed
        B = self._hmm.output_probabilities
        C = np.dot(np.dot(B.T, Pi_c),np.dot(P_c_k, B))
        P = C / C[:,None] # row normalization
        return P

    @property
    def stationary_distribution(self):
        # TODO: same discussion as in transition_matrix, above
        raise NotImplementedError('What do we do here?')

    @property
    def metastable_stationary_distribution(self):
        # TODO: Or hidden_transition_matrix? Or transition_matrix_hidden?
        return self._hmm.transition_matrix

    @property
    def obs_stationary_distribution(self):
        return np.dot(self._hmm.stationary_distribution, self._hmm.output_probabilities)

    @property
    def eigenvalues(self):
        return self._hmm.eigenvalues

    # TODO: What if we just get either a coarse-grained or a fine-grained MSM at request?
    # TODO: That would avoid some naming confusion.

    @property
    def timescales(self):
        return self._hmm.timescales

    @property
    def metastable_eigenvectors_left(self):
        return self._hmm.eigenvectors_left

    @property
    def metastable_eigenvectors_right(self):
        return self._hmm.eigenvectors_right

    @property
    def obs_eigenvectors_left(self):
        return np.dot(self._hmm.eigenvectors_left, self._hmm.output_probabilities)

    @property
    def obs_eigenvectors_right(self):
        return np.dot(self._hmm.output_probabilities.T, self._hmm.eigenvectors_right)

    # ==================================================================================================================
    # The rest is simply implemented by directly analyzing the observed transition matrix
    # TODO: maybe we should save P_obs if we use it for many functions
    #
    # TODO: this is a lot of duplicate code. We can generalize it. Models can be generalized to providing
    # TODO: Eigenvalues, eigenvectors and some also a transition matrix. The calculations can then be made by
    # TODO: Algorithms which just receive this information.

    def _assert_in_active(self, A):
        """
        Checks if set A is within the set of observed states

        Parameters
        ----------
        A : int or int array
            set of states
        """
        assert np.max(A) < self.nstates, 'Chosen set contains states that are not included in the active set.'

    def _mfpt(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import mfpt as __mfpt
        # scale mfpt by lag time
        return self._hmm.lag * __mfpt(P, B, origin=A, mu=mu)

    def mfpt(self, A, B):
        return self._mfpt(self.obs_transition_matrix(1), A, B, mu=self.obs_stationary_distribution)

    def _committor_forward(self, P, A, B):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as __committor
        return __committor(self.obs_transition_matrix(1), A, B, forward=True)

    def committor_forward(self, A, B):
        return self._committor_forward(self.obs_transition_matrix(1), A, B)

    def _committor_backward(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as __committor
        return __committor(P, A, B, forward=False, mu=mu)

    def committor_backward(self, A, B):
        return self._committor_backward(self.obs_transition_matrix(1), A,B, mu=self.stationary_distribution)

    def expectation(self, a):
        assert np.shape(a)[0] == self.nstates, \
            'observable vector a does not have same size like the active set. '+ 'Need len(a) = ' + str(self.nstates)
        return np.dot(a, self.obs_stationary_distribution)


    # TODO: First project to metastable, then compute correlation etc for small system.

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        # check input
        assert np.shape(a)[0] == self.nstates, \
            'observable vector a does not have same size like the active set. Need len(a) = ' + str(self.nstates)
        if b is not None:
            assert np.shape(b)[0] == self.nstates, \
                'observable vector b does not have same size like the active set. Need len(b) = ' + str(self.nstates)
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        kmax = int(ceil(float(maxtime) / self.lag))
        steps = np.array(range(kmax), dtype=int)
        # compute correlation
        from pyemma.msm.analysis import correlation as _correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _correlation(self.obs_transition_matrix(1), a, obs2=b, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self.lag * steps
        return times, res

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        # will not compute for nonreversible matrices
        if (not self.is_reversible) and (self.nstates > 2):
            raise ValueError('Fingerprint calculation is not supported for nonreversible transition matrices. '+
                             'Consider estimating the MSM with reversible = True')
        # check input
        assert np.shape(a)[0] == self.nstates, \
            'observable vector a does not have same size like the active set. Need len(a) = ' + str(self.nstates)
        if b is not None:
            assert np.shape(b)[0] == self.nstates, \
                'observable vector b does not have same size like the active set. Need len(b) = ' + str(self.nstates)
        from pyemma.msm.analysis import fingerprint_correlation as _fc
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _fc(self.obs_transition_matrix(1), a, obs2=b, tau=self.lag, k=k, ncv=ncv)

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        # check input
        assert np.shape(p0)[0] == self.nstates, \
            'initial distribution p0 does not have same size like the active set. Need len(p0) = ' + str(self.nstates)
        assert np.shape(a)[0] == self.nstates, \
            'observable vector a does not have same size like the active set. Need len(a) = ' + str(self.nstates)
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        kmax = int(ceil(float(maxtime) / self.lag))
        steps = np.array(range(kmax), dtype=int)
        # compute relaxation function
        from pyemma.msm.analysis import relaxation as _relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _relaxation(self.obs_transition_matrix(1), p0, a, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self.lag * steps
        return times, res

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        # will not compute for nonreversible matrices
        if (not self.is_reversible) and (self.nstates > 2):
            raise ValueError('Fingerprint calculation is not supported for nonreversible transition matrices. '+
                             'Consider estimating the MSM with reversible = True')
        # check input
        assert np.shape(p0)[0] == self.nstates, \
            'initial distribution p0 does not have same size like the active set. Need len(p0) = ' + str(self.nstates)
        assert np.shape(a)[0] == self.nstates, \
            'observable vector a does not have the same size like the active set. Need len(a) = ' + str(self.nstates)
        from pyemma.msm.analysis import fingerprint_relaxation as _fr
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _fr(self.obs_transition_matrix(1), p0, a, tau=self.lag, k=k, ncv=ncv)


