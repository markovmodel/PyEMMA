
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

"""

__docformat__ = "restructuredtext en"

import numpy as _np
from pyemma.msm.ui.msm import MSM as _MSM
from pyemma.util import types as _types
from pyemma.util.annotators import shortcut

class HMSM(_MSM):
    r""" Hidden Markov model on discrete states.

    Parameters
    ----------
    hmm : :class:`DiscreteHMM <bhmm.DiscreteHMM>`
        Hidden Markov Model

    """

    def __init__(self, Pcoarse, Pobs, dt='1 step'):
        """

        Parameters
        ----------
        Pcoarse : ndarray (m,m)
            coarse-grained or hidden transition matrix
        Pobs : ndarray (m,n)
            observation probability matrix from hidden to observable discrete states
        dt : str, optional, default='1 step'
            time step of the model

        """
        # construct superclass and check input
        _MSM.__init__(self, Pcoarse, dt)
        # check and save copy of output probability
        assert _types.is_float_matrix(Pobs), 'Pout is not a matrix of floating numbers'
        self._Pobs = _np.array(Pobs)
        assert _np.allclose(self._Pobs.sum(axis=1), 1), 'Pout is not a stochastic matrix'

    @property
    def nstates_obs(self):
        return self._Pobs.shape[1]

    @property
    def observation_probabilities(self):
        r""" returns the output probability matrix

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        """
        return self._Pobs

    def transition_matrix_obs(self, k=1):
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
        Pi_c = _np.diag(self.stationary_distribution)
        P_c = self.transition_matrix
        P_c_k = _np.linalg.matrix_power(P_c, k) # take a power if needed
        B = self._Pobs
        C = _np.dot(_np.dot(B.T, Pi_c),_np.dot(P_c_k, B))
        P = C / C.sum(axis=1)[:,None] # row normalization
        return P

    @property
    def stationary_distribution_obs(self):
        return _np.dot(self.stationary_distribution, self._Pobs)

    @property
    def eigenvectors_left_obs(self):
        return _np.dot(self.eigenvectors_left(), self._Pobs)

    @property
    def eigenvectors_right_obs(self):
        return _np.dot(self.metastable_memberships, self.eigenvectors_right())

    # ================================================================================================================
    # Experimental properties: Here we allow to use either coarse-grained or microstate observables
    # ================================================================================================================

    def expectation(self, a):
        a = _types.ensure_float_vector(a, require_order=True)
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            # project to hidden and compute
            a = _np.dot(self._Pobs, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).expectation(a)
        else:
            raise ValueError('observable vector a has size '+len(a)+' which is incompatible with both hidden ('+
                             self.nstates+') and observed states ('+self.nstates_obs+')')

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        a = _types.ensure_ndarray(a, ndim=1, kind='numeric')
        b = _types.ensure_ndarray_or_None(b, ndim=1, kind='numeric', size=len(a))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            a = _np.dot(self._Pobs, a)
            if b is not None:
                b = _np.dot(self._Pobs, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).correlation(a, b=b, maxtime=maxtime)
        else:
            raise ValueError('observable vectors have size '+len(a)+' which is incompatible with both hidden ('+
                             self.nstates+') and observed states ('+self.nstates_obs+')')

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        # basic checks for a and b
        a = _types.ensure_ndarray(a, ndim=1, kind='numeric')
        b = _types.ensure_ndarray_or_None(b, ndim=1, kind='numeric', size=len(a))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            a = _np.dot(self._Pobs, a)
            if b is not None:
                b = _np.dot(self._Pobs, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).fingerprint_correlation(a, b=b)
        else:
            raise ValueError('observable vectors have size '+len(a)+' which is incompatible with both hidden ('+
                             self.nstates+') and observed states ('+self.nstates_obs+')')

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        p0 = _types.ensure_ndarray(p0, ndim=1, kind='numeric')
        a = _types.ensure_ndarray(a, ndim=1, kind='numeric', size=len(p0))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            p0 = _np.dot(self._Pobs, p0)
            a = _np.dot(self._Pobs, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).relaxation(p0, a, maxtime=maxtime)
        else:
            raise ValueError('observable vectors have size '+len(a)+' which is incompatible with both hidden ('+
                             self.nstates+') and observed states ('+self.nstates_obs+')')

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        # basic checks for a and b
        p0 = _types.ensure_ndarray(p0, ndim=1, kind='numeric')
        a = _types.ensure_ndarray(a, ndim=1, kind='numeric', size=len(p0))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            p0 = _np.dot(self._Pobs, p0)
            a = _np.dot(self._Pobs, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).fingerprint_relaxation(p0, a)
        else:
            raise ValueError('observable vectors have size '+len(a)+' which is incompatible with both hidden ('+
                             self.nstates+') and observed states ('+self.nstates_obs+')')

    def pcca(self, m):
        raise NotImplementedError('PCCA is not meaningful for Hidden Markov models. '+
                                  'If you really want to do this, initialize an MSM with the HMSM transition matrix.')

    # ================================================================================================================
    # Metastable state stuff is overwritten, because we now have the HMM output probability matrix
    # ================================================================================================================

    @property
    def metastable_memberships(self):
        """ Computes the memberships of observable states to metastable sets by Bayesian inversion as described in [1]_.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each observable state to be assigned to each
            metastable or hidden state. The row sums of M are 1.

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        A = _np.dot(_np.diag(self.stationary_distribution), self._Pobs)
        M = _np.dot(A, _np.diag(1.0/self.stationary_distribution_obs)).T
        # renormalize
        M /= M.sum(axis=1)[:,None]
        return M

    @property
    def metastable_distributions(self):
        """ Returns the output probability distributions. Identical to :meth:`observation_probability`

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        See also
        --------
        observation_probability

        """
        return self._Pobs

    @property
    def metastable_sets(self):
        """ Computes the metastable sets of observable states within each metastable set

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        A list of length equal to metastable states. Each element is an array with observable state indexes contained
        in it

        """
        res = []
        assignment = self.metastable_assignments
        for i in range(self.nstates):
            res.append(_np.where(assignment == i)[0])
        return res

    @property
    def metastable_assignments(self):
        """ Computes the assignment to metastable sets for observable states

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        For each observable state, the metastable state it is located in.

        """
        return _np.argmax(self._Pobs, axis=0)

