
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

from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma._base.model import SampledModel as _SampledModel
from pyemma.util.types import is_iterable


class SampledHMSM(_HMSM, _SampledModel):

    def __init__(self, samples, ref=None, conf=0.95):
        r""" Constructs a sampled HMSM

        Parameters
        ----------
        samples : list of HMSM
            Sampled HMSM objects

        ref : HMSM
            Single-point estimator, e.g. containing a maximum likelihood HMSM.
            If not given, the sample mean will be used.

        conf : float, optional, default=0.95
            Confidence interval. By default two-sigma (95.4%) is used.
            Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # validate input
        assert is_iterable(samples), 'samples must be a list of MSM objects, but is not.'
        assert isinstance(samples[0], _HMSM), 'samples must be a list of MSM objects, but is not.'
        # construct superclass 1
        _SampledModel.__init__(self, samples, conf=conf)
        # construct superclass 2
        if ref is None:
            Pref = self.sample_mean('P')
            pobsref = self.sample_mean('pobs')
            _HMSM.__init__(self, Pref, pobsref, dt_model=samples[0].dt_model)
        else:
            _HMSM.__init__(self, ref.Pref, ref.pobs, dt_model=ref.dt_model)


    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, samples=None, conf=0.95,
                         P=None, pobs=None, pi=None, reversible=None, dt_model='1 step', neig=None):
        """

        Parameters
        ----------
        samples : list of MSM objects
            sampled MSMs
        conf : float, optional, default=0.68
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # set model parameters of superclass
        _SampledModel.set_model_params(self, samples=samples, conf=conf)
        _HMSM.set_model_params(self, P=P, pobs=pobs, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)


    # def _do_sample_eigendecomposition(self):
    #     """Conducts the eigenvalue decompositions for all sampled matrices.
    #
    #     Stores all eigenvalues, left and right eigenvectors for all sampled matrices
    #
    #     """
    #     from pyemma.msm.analysis import rdl_decomposition
    #     from pyemma.util import linalg
    #
    #     # left eigenvectors
    #     self._sample_Ls = _np.empty((self._nsamples, self._nstates, self._nstates), dtype=float)
    #     # eigenvalues
    #     self._sample_eigenvalues = _np.empty((self._nsamples, self._nstates), dtype=float)
    #     # right eigenvectors
    #     self._sample_Rs = _np.empty((self._nsamples, self._nstates, self._nstates), dtype=float)
    #
    #     for i in range(self._nsamples):
    #         if self._reversible:
    #             R, D, L = rdl_decomposition(self._sample_Ps[i], norm='reversible')
    #             # everything must be real-valued
    #             R = R.real
    #             D = D.real
    #             L = L.real
    #         else:
    #             R, D, L = rdl_decomposition(self._sample_Ps[i], norm='standard')
    #         # assign ordered
    #         I = linalg.match_eigenvectors(self.eigenvectors_right, R,
    #                                       w_ref=self.stationary_distribution, w=self._sample_mus[i])
    #         self._sample_Ls[i, :, :] = L[I, :]
    #         self._sample_eigenvalues[i, :] = _np.diag(D)[I]
    #         self._sample_Rs[i, :, :] = R[:, I]
    #
    # def set_confidence(self, conf):
    #     self._confidence = conf
    #
    # @property
    # def nsamples(self):
    #     r""" Number of samples """
    #     return self._nsamples
    #
    # @property
    # def confidence_interval(self):
    #     r""" Confidence interval used """
    #     return self._confidence
    #
    # @property
    # def stationary_distribution_samples(self):
    #     r""" Samples of the initial distribution """
    #     return self._sample_mus
    #
    # @property
    # def stationary_distribution_mean(self):
    #     r""" The mean of the initial distribution of the hidden states """
    #     return _np.mean(self.stationary_distribution_samples, axis=0)
    #
    # @property
    # def stationary_distribution_std(self):
    #     r""" The standard deviation of the initial distribution of the hidden states """
    #     return _np.std(self.stationary_distribution_samples, axis=0)
    #
    # @property
    # def stationary_distribution_conf(self):
    #     r""" The confidence interval of the initial distribution of the hidden states """
    #     return confidence_interval(self.stationary_distribution_samples, alpha=self._confidence)
    #
    # @property
    # def transition_matrix_samples(self):
    #     r""" Samples of the transition matrix """
    #     return self._sample_Ps
    #
    # @property
    # def transition_matrix_mean(self):
    #     r""" The mean of the transition_matrix of the hidden states """
    #     return _np.mean(self.transition_matrix_samples, axis=0)
    #
    # @property
    # def transition_matrix_std(self):
    #     r""" The standard deviation of the transition_matrix of the hidden states """
    #     return _np.std(self.transition_matrix_samples, axis=0)
    #
    # @property
    # def transition_matrix_conf(self):
    #     r""" The confidence interval of the transition_matrix of the hidden states """
    #     return confidence_interval(self.transition_matrix_samples, alpha=self._confidence)
    #
    # @property
    # def output_probabilities_samples(self):
    #     r""" Samples of the output probability matrix """
    #     return self._sample_pobs
    #
    # @property
    # def output_probabilities_mean(self):
    #     r""" The mean of the output probability matrix """
    #     return _np.mean(self.output_probabilities_samples, axis=0)
    #
    # @property
    # def output_probabilities_std(self):
    #     r""" The standard deviation of the output probability matrix """
    #     return _np.std(self.output_probabilities_samples, axis=0)
    #
    # @property
    # def output_probabilities_conf(self):
    #     r""" The standard deviation of the output probability matrix """
    #     return confidence_interval(self.output_probabilities_samples, alpha=self._confidence)
    #
    # @property
    # def eigenvalues_samples(self):
    #     r""" Samples of the eigenvalues """
    #     return self._sample_eigenvalues
    #
    # @property
    # def eigenvalues_mean(self):
    #     r""" The mean of the eigenvalues of the hidden states """
    #     return _np.mean(self.eigenvalues_samples, axis=0)
    #
    # @property
    # def eigenvalues_std(self):
    #     r""" The standard deviation of the eigenvalues of the hidden states """
    #     return _np.std(self.eigenvalues_samples, axis=0)
    #
    # @property
    # def eigenvalues_conf(self):
    #     r""" The confidence interval of the eigenvalues of the hidden states """
    #     return confidence_interval(self.eigenvalues_samples, alpha=self._confidence)
    #
    # @property
    # def eigenvectors_left_samples(self):
    #     r""" Samples of the left eigenvectors of the hidden transition matrix """
    #     return self._sample_Ls
    #
    # @property
    # def eigenvectors_left_mean(self):
    #     r""" The mean of the left eigenvectors of the hidden transition matrix """
    #     return _np.mean(self.eigenvectors_left_samples, axis=0)
    #
    # @property
    # def eigenvectors_left_std(self):
    #     r""" The standard deviation of the left eigenvectors of the hidden transition matrix """
    #     return _np.std(self.eigenvectors_left_samples, axis=0)
    #
    # @property
    # def eigenvectors_left_conf(self):
    #     r""" The confidence interval of the left eigenvectors of the hidden transition matrix """
    #     return confidence_interval(self.eigenvectors_left_samples, alpha=self._confidence)
    #
    # @property
    # def eigenvectors_right_samples(self):
    #     r""" Samples of the right eigenvectors of the hidden transition matrix """
    #     return self._sample_Rs
    #
    # @property
    # def eigenvectors_right_mean(self):
    #     r""" The mean of the right eigenvectors of the hidden transition matrix """
    #     return _np.mean(self.eigenvectors_right_samples, axis=0)
    #
    # @property
    # def eigenvectors_right_std(self):
    #     r""" The standard deviation of the right eigenvectors of the hidden transition matrix """
    #     return _np.std(self.eigenvectors_right_samples, axis=0)
    #
    # @property
    # def eigenvectors_right_conf(self):
    #     r""" The confidence interval of the right eigenvectors of the hidden transition matrix """
    #     return confidence_interval(self.eigenvectors_right_samples, alpha=self._confidence)
    #
    # @property
    # def timescales_samples(self):
    #     r""" Samples of the timescales """
    #     return -self.lagtime / _np.log(_np.abs(self._sample_eigenvalues[:,1:]))
    #
    # @property
    # def timescales_mean(self):
    #     r""" The mean of the timescales of the hidden states """
    #     return _np.mean(self.timescales_samples, axis=0)
    #
    # @property
    # def timescales_std(self):
    #     r""" The standard deviation of the timescales of the hidden states """
    #     return _np.std(self.timescales_samples, axis=0)
    #
    # @property
    # def timescales_conf(self):
    #     r""" The confidence interval of the timescales of the hidden states """
    #     return confidence_interval(self.timescales_samples, alpha=self._confidence)
    #
    # @property
    # def lifetimes_samples(self):
    #     r""" Samples of the lifetimes """
    #     res = _np.empty((self.nsamples, self.nstates), dtype=float)
    #     for i in range(self.nsamples):
    #         res[i,:] = -self._lag / _np.log(_np.diag(self._sample_Ps[i]))
    #     return res
    #
    # @property
    # def lifetimes_mean(self):
    #     r""" The mean of the lifetimes of the hidden states """
    #     return _np.mean(self.lifetimes_samples, axis=0)
    #
    # @property
    # def lifetimes_std(self):
    #     r""" The standard deviation of the lifetimes of the hidden states """
    #     return _np.std(self.lifetimes_samples, axis=0)
    #
    # @property
    # def lifetimes_conf(self):
    #     r""" The confidence interval of the lifetimes of the hidden states """
    #     return confidence_interval(self.lifetimes_samples, alpha=self._confidence)
