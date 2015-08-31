
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

from __future__ import absolute_import
__author__ = 'noe'

from pyemma._base.model import SampledModel
from pyemma.msm.models.msm import MSM
from pyemma.util.types import is_iterable

class SampledMSM(MSM, SampledModel):
    r""" Sampled Markov state model """

    def __init__(self, samples, ref=None, conf=0.95):
        r""" Constructs a sampled MSM

        Parameters
        ----------
        samples : list of MSM
            Sampled MSM objects
        ref : EstimatedMSM
            Single-point estimator, e.g. containing a maximum likelihood or mean MSM
        conf : float, optional, default=0.95
            Confidence interval. By default two-sigma (95.4%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # validate input
        assert is_iterable(samples), 'samples must be a list of MSM objects, but is not.'
        assert isinstance(samples[0], MSM), 'samples must be a list of MSM objects, but is not.'
        # construct superclass 1
        SampledModel.__init__(self, samples, conf=conf)
        # construct superclass 2
        if ref is None:
            Pref = self.sample_mean('P')
            MSM.__init__(self, Pref, dt_model=samples[0].dt_model, neig=samples[0].neig, ncv=samples[0].ncv)
        else:
            MSM.__init__(self, ref.Pref, pi=ref.pi, reversible=ref.reversible, dt_model=ref.dt_model,
                         neig=ref.neig, ncv=ref.ncv)

    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, samples=None, conf=0.95,
                         P=None, pi=None, reversible=None, dt_model='1 step', neig=None):
        """

        Parameters
        ----------
        samples : list of MSM objects
            sampled MSMs
        conf : float, optional, default=0.68
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # set model parameters of superclass
        SampledModel.set_model_params(self, samples=samples, conf=conf)
        MSM.set_model_params(self, P=P, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)


#
# class SampledEstimatedMSM(EstimatedMSM, SampledModel):
#
#     def __init__(self, samples, ref, Pref='mle', conf=0.95):
#         r""" Constructs a sampled MSM
#
#         Parameters
#         ----------
#         samples : list of MSM
#             Sampled MSM objects
#         ref : EstimatedMSM
#             Single-point estimator, e.g. containing a maximum likelihood or mean MSM
#         conf : float, optional, default=0.68
#             Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.
#
#         """
#         # construct superclass 1
#         SampledModel.__init__(self, samples, conf=conf)
#         # use reference or mean MSM.
#         if ref is None:
#             Pref = self.sample_mean('P')
#         else:
#             Pref = ref.P
#         # construct superclass 2
#         EstimatedMSM.__init__(self, ref.discrete_trajectories_full, ref.timestep, ref.lagtime, ref.connectivity,
#                               ref.active_set, ref.connected_sets, ref.count_matrix_full, ref.count_matrix_active, Pref)


#     def _do_sample_eigendecomposition(self, k, ncv=None):
#         """Conducts the eigenvalue decompositions for all sampled matrices.
#
#         Stores k eigenvalues, left and right eigenvectors for all sampled matrices
#
#         Parameters
#         ----------
#         k : int
#             The number of eigenvalues / eigenvectors to be kept
#         ncv : int (optional)
#             Relevant for eigenvalue decomposition of reversible transition matrices.
#             ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
#             it is recommended that ncv > 2*k
#
#         """
#         from msmtools.analysis import rdl_decomposition
#         from pyemma.util import linalg
#
#         # left eigenvectors
#         self.sample_Ls = np.empty((self._nsample), dtype=object)
#         # eigenvalues
#         self.sample_eigenvalues = np.empty((self._nsample), dtype=object)
#         # right eigenvectors
#         self.sample_Rs = np.empty((self._nsample), dtype=object)
#         # eigenvector assignments
#         self.sample_eig_assignments = np.empty((self._nsample), dtype=object)
#
#         for i in range(self._nsample):
#             if self._reversible:
#                 R, D, L = rdl_decomposition(self.sample_Ps[i], k=k, norm='reversible', ncv=ncv)
#                 # everything must be real-valued
#                 R = R.real
#                 D = D.real
#                 L = L.real
#             else:
#                 R, D, L = rdl_decomposition(self.sample_Ps[i], k=k, norm='standard', ncv=ncv)
#             # assign ordered
#             I = linalg.match_eigenvectors(self.eigenvectors_right(), R,
#                                           w_ref=self.stationary_distribution, w=self.sample_mus[i])
#             self.sample_Ls[i] = L[I,:]
#             self.sample_eigenvalues[i] = np.diag(D)[I]
#             self.sample_Rs[i] = R[:,I]
#
#     def _ensure_sample_eigendecomposition(self, k=None, ncv=None):
#         """Ensures that eigendecomposition has been performed with at least k eigenpairs
#
#         k : int
#             number of eigenpairs needed. This setting is mandatory for sparse transition matrices
#             (if you set sparse=True in the initialization). For dense matrices, k will be ignored
#             as all eigenvalues and eigenvectors will be computed and stored.
#         ncv : int (optional)
#             Relevant for eigenvalue decomposition of reversible transition matrices.
#             ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
#             it is recommended that ncv > 2*k
#
#         """
#         # check input?
#         if self._sparse:
#             if k is None:
#                 raise ValueError(
#                     'You have requested sparse=True, then the number of eigenvalues neig must also be set.')
#         else:
#             # override setting - we anyway have to compute all eigenvalues, so we'll also store them.
#             k = self._nstates
#         # ensure that eigenvalue decomposition with k components is done.
#         try:
#             m = len(self.sample_eigenvalues[0])  # this will raise and exception if self._eigenvalues doesn't exist yet.
#             if m < k:
#                 # not enough eigenpairs present - recompute:
#                 self._do_sample_eigendecomposition(k, ncv=ncv)
#         except:
#             # no eigendecomposition yet - compute:
#             self._do_sample_eigendecomposition(k, ncv=ncv)
#
#     @property
#     def stationary_distribution_mean(self):
#         """Sample mean for the stationary distribution on the active set.
#
#         See also
#         --------
#         MSM.stationary_distribution
#
#         """
#         return np.mean(self.sample_mus, axis=0)
#
#     @property
#     def stationary_distribution_std(self):
#         """Sample standard deviation for the stationary distribution on the active set.
#
#         See also
#         --------
#         MSM.stationary_distribution
#
#         """
#         return np.std(self.sample_mus, axis=0)
#
#     @property
#     def stationary_distribution_conf(self):
#         """Sample confidence interval for the stationary distribution on the active set.
#
#         See also
#         --------
#         MSM.stationary_distribution
#
#         """
#         return stat.confidence_interval(self.sample_mus, alpha=self._confidence)
#
#     def eigenvalues_mean(self, k=None, ncv=None):
#         """Sample mean for the eigenvalues.
#
#         See also
#         --------
#         MSM.eigenvalues
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.mean(self.sample_eigenvalues, axis=0)
#
#     def eigenvalues_std(self, k=None, ncv=None):
#         """Sample standard deviation for the eigenvalues.
#
#         See also
#         --------
#         MSM.eigenvalues
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.std(self.sample_eigenvalues, axis=0)
#
#     def eigenvalues_conf(self, k=None, ncv=None):
#         """Sample confidence interval for the eigenvalues.
#
#         See also
#         --------
#         MSM.eigenvalues
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return stat.confidence_interval(self.sample_eigenvalues, alpha=self._confidence)
#
#     def eigenvectors_left_mean(self, k=None, ncv=None):
#         """Sample mean for the left eigenvectors.
#
#         See also
#         --------
#         MSM.eigenvectors_left
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.mean(self.sample_Ls, axis=0)
#
#     def eigenvectors_left_std(self, k=None, ncv=None):
#         """Sample standard deviation for the left eigenvectors.
#
#         See also
#         --------
#         MSM.eigenvectors_left
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.std(self.sample_Ls, axis=0)
#
#     def eigenvectors_left_conf(self, k=None, ncv=None):
#         """Sample confidence interval for the left eigenvectors.
#
#         See also
#         --------
#         MSM.eigenvectors_left
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return stat.confidence_interval(self.sample_Ls, alpha=self._confidence)
#
#
# #     def eigenvectors_right_mean(self, k=None, ncv=None):
# #         """Sample mean for the right eigenvectors.
# #
# #         See also
# #         --------
# #         MSM.eigenvectors_right
# #
# #         """
# #         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
# #         return np.mean(self.sample_Rs, axis=0)
# #
# #     def eigenvectors_right_std(self, k=None, ncv=None):
# #         """Sample standard deviation for the right eigenvectors.
# #
# #         See also
# #         --------
# #         MSM.eigenvectors_right
# #
# #         """
# #         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
# #         return np.std(self.sample_Rs, axis=0)
# #
# #     def eigenvectors_right_conf(self, k=None, ncv=None):
# #         """Sample confidence interval for the right eigenvectors.
# #
# #         See also
# #         --------
# #         MSM.eigenvectors_right
# #
# #         """
# #         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
# #         return stat.confidence_interval_arr(self.sample_Rs, alpha=self._confidence)
#
#     def _sample_timescales(self):
#         """Compute sample timescales from the sample eigenvalues"""
#         res = np.empty((self._nsample), dtype=np.object)
#         for i in range(self._nsample):
#             res[i] = -self._lag / np.log(np.abs(self.sample_eigenvalues[i][1:]))
#         return res
#
#     def timescales_mean(self, k=None, ncv=None):
#         """Sample mean for the timescales.
#
#         See also
#         --------
#         MSM.timescales
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.mean(self._sample_timescales(), axis=0)
#
#     def timescales_std(self, k=None, ncv=None):
#         """Sample standard deviation for the timescales.
#
#         See also
#         --------
#         MSM.timescales
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return np.std(self._sample_timescales(), axis=0)
#
#     def timescales_conf(self, k=None, ncv=None):
#         """Sample confidence interval for the timescales.
#
#         See also
#         --------
#         MSM.timescales
#
#         """
#         self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
#         return stat.confidence_interval(self._sample_timescales(), alpha=self._confidence)
#
#
#     def _sample_mfpt(self, A, B):
#         """Compute sample timescales from the sample eigenvalues"""
#         res = np.zeros((self._nsample))
#         for i in range(self._nsample):
#             res[i] = self._mfpt(self.sample_Ps[i], A, B, mu=self.sample_mus[i])
#         return res
#
#     def mfpt_mean(self, A, B):
#         """Sample mean for the A->B mean first passage time.
#
#         See also
#         --------
#         MSM.mfpt
#
#         """
#         return np.mean(self._sample_mfpt(A,B), axis=0)
#
#     def mfpt_std(self, A, B):
#         """Sample standard deviation for the A->B mean first passage time.
#
#         See also
#         --------
#         MSM.mfpt
#
#         """
#         return np.std(self._sample_mfpt(A,B), axis=0)
#
#     def mfpt_conf(self, A, B):
#         """Sample confidence interval for the A->B mean first passage time.
#
#         See also
#         --------
#         MSM.mfpt
#
#         """
#         return stat.confidence_interval(self._sample_mfpt(A,B), alpha=self._confidence)
#
#     def _sample_committor_forward(self, A, B):
#         """Compute sample timescales from the sample eigenvalues"""
#         res = np.empty((self._nsample), dtype=np.object)
#         for i in range(self._nsample):
#             res[i] = self._committor_forward(self.sample_Ps[i], A, B)
#         return res
#
#     def committor_forward_mean(self, A, B):
#         """Sample mean for the A->B forward committor.
#
#         See also
#         --------
#         MSM.committor_forward
#
#         """
#         return np.mean(self._sample_committor_forward(A,B), axis=0)
#
#     def committor_forward_std(self, A, B):
#         """Sample standard deviation for the A->B forward committor.
#
#         See also
#         --------
#         MSM.committor_forward
#
#         """
#         return np.std(self._sample_committor_forward(A,B), axis=0)
#
#     def committor_forward_conf(self, A, B):
#         """Sample confidence interval for the A->B forward committor.
#
#         See also
#         --------
#         MSM.committor_forward
#
#         """
#         return stat.confidence_interval(self._sample_committor_forward(A,B), alpha=self._confidence)
#
#
#     def _sample_committor_backward(self, A, B):
#         """Compute sample timescales from the sample eigenvalues"""
#         res = np.empty((self._nsample), dtype=np.object)
#         for i in range(self._nsample):
#             res[i] = self._committor_backward(self.sample_Ps[i], A, B, mu=self.sample_mus[i])
#         return res
#
#     def committor_backward_mean(self, A, B):
#         """Sample mean for the A->B backward committor.
#
#         See also
#         --------
#         MSM.committor_backward
#
#         """
#         return np.mean(self._sample_committor_backward(A,B), axis=0)
#
#     def committor_backward_std(self, A, B):
#         """Sample standard deviation for the A->B backward committor.
#
#         See also
#         --------
#         MSM.committor_backward
#
#         """
#         return np.std(self._sample_committor_backward(A,B), axis=0)
#
#     def committor_backward_conf(self, A, B):
#         """Sample confidence interval for the A->B backward committor.
#
#         See also
#         --------
#         MSM.committor_backward
#
#         """
#         return stat.confidence_interval(self._sample_committor_backward(A,B), alpha=self._confidence)