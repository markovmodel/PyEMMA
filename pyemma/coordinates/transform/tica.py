
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


'''
Created on 19.01.2015

@author: marscher
'''

from __future__ import absolute_import

from pyemma.util import types
from pyemma.util.annotators import doc_inherit
from pyemma.util.linalg import eig_corr
from pyemma.util.reflection import get_default_args

import numpy as np

from .transformer import Transformer, SkipPassException


__all__ = ['TICA']


class MeaningOfLagWithStrideWarning(UserWarning):
    pass


class TICA(Transformer):
    r""" Time-lagged independent component analysis (TICA)"""

    def __init__(self, lag, dim=-1, var_cutoff=0.95, kinetic_map=True, epsilon=1e-6,
                 force_eigenvalues_le_one=False, mean=None):
        r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
        tau : int
            lag time
        dim : int, optional, default -1
            Maximum number of significant independent components to use to reduce dimension of input data. -1 means
            all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.
        var_cutoff : float in the range [0,1], optional, default 0.95
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim
        kinetic_map : bool, optional, default True
            Eigenvectors will be scaled by eigenvalues. As a result, Euclidean distances in the transformed data
            approximate kinetic distances [4]_. This is a good choice when the data is further processed by clustering.
        epsilon : float
            eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
        force_eigenvalues_le_one : boolean
            Compute covariance matrix and time-lagged covariance matrix such
            that the generalized eigenvalues are always guaranteed to be <= 1.
        mean : ndarray, optional, default None
            Optionally pass pre-calculated means to avoid their re-computation.
            The shape has to match the input dimension.

        Notes
        -----
        Given a sequence of multivariate data :math:`X_t`, computes the mean-free
        covariance and time-lagged covariance matrix:

        .. math::

            C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
            C_{\tau} &= (X_t - \mu)^T (X_{t + \tau} - \mu)

        and solves the eigenvalue problem

        .. math:: C_{\tau} r_i = C_0 \lambda_i(tau) r_i,

        where :math:`r_i` are the independent components and :math:`\lambda_i(tau)` are
        their respective normalized time-autocorrelations. The eigenvalues are
        related to the relaxation timescale by

        .. math:: t_i(tau) = -\tau / \ln |\lambda_i|.

        When used as a dimension reduction method, the input data is projected
        onto the dominant independent components.

        References
        ----------
        .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
           Identification of slow molecular order parameters for Markov model construction
           J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489
        .. [2] Schwantes C, V S Pande. 2013.
           Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
           J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a
        .. [3] L. Molgedey and H. G. Schuster. 1994.
           Separation of a mixture of independent signals using time delayed correlations
           Phys. Rev. Lett. 72, 3634.
        .. [4] Noe, F. and C. Clementi. 2015.
            Kinetic distance and kinetic maps from molecular dynamics simulation
            http://arxiv.org/abs/1506.06259

        """
        super(TICA, self).__init__()

        # store lag time to set it appropriately in second pass of parametrize
        self._lag = lag
        self._dim = dim
        self._var_cutoff = var_cutoff
        default_var_cutoff = get_default_args(self.__init__)['var_cutoff']
        if dim != -1 and var_cutoff != default_var_cutoff:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')
        self._kinetic_map = kinetic_map
        self._epsilon = epsilon
        self._force_eigenvalues_le_one = force_eigenvalues_le_one

        # covariances
        self.cov = None
        self.cov_tau = None
        # mean
        self.mu = mean

        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0
        self._eigenvalues = None
        self._eigenvectors = None
        self._cumvar = None

        self._custom_param_progress_handling = True

        # skipped trajectories
        self._skipped_trajs = []

    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        raise Exception('Changing lag time is not allowed at the moment.')
        #self._parametrized = False
        #self._lag = new_tau

    @doc_inherit
    def describe(self):
        dim = self._dim
        try:
            dim = self.dimension()
        except:
            pass
        return "[TICA, lag = %i; max. output dim. = %i]" % (self._lag, dim)

    def dimension(self):
        """ output dimension """
        d = None
        if self._dim != -1:  # fixed parametrization
            d = self._dim
        elif self._parametrized:  # parametrization finished. Dimension is known
            dim = len(self._eigenvalues)
            if self._var_cutoff < 1.0:  # if subspace_variance, reduce the output dimension if needed
                dim = min(dim, np.searchsorted(self._cumvar, self._var_cutoff)+1)
            d = dim
        elif self._var_cutoff == 1.0:  # We only know that all dimensions are wanted, so return input dim
            d = self.data_producer.dimension()
        else:  # We know nothing. Give up
            raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                               'transformer has not yet been parametrized. Call parametrize() before.')
        return d

    @property
    def mean(self):
        """ mean of input features """
        return self.mu

    def _param_init(self):
        indim = self.data_producer.dimension()
        assert indim > 0, "zero dimension from data producer"
        assert self._dim <= indim, ("requested more output dimensions (%i) than dimension"
                                    " of input data (%i)" % (self._dim, indim))
        if self._force_eigenvalues_le_one and self._lag % self._param_with_stride != 0:
            raise RuntimeError("When using TICA with force_eigenvalues_le_one, lag must be a multiple of stride.")

        if self.mu is not None:
            self.mu = types.ensure_ndarray(self.mu, shape=(indim,))
            self._given_mean = True
        else:
            self.mu = np.zeros(indim)
            self._given_mean = False

        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0

        self._logger.debug("Running TICA with tau=%i)" % self._lag)

        # amount of chunks
        denom = self._n_chunks(self._param_with_stride)
        self._progress_register(denom, "calculate mean", 0)
        self._progress_register(denom, "calculate covariances", 1)

        n = self.data_producer.dimension()
        if __debug__:
            self._logger.debug("create cov matrices, %i x %i" % (n, n))

        self.cov = np.zeros((n, n))
        self.cov_tau = np.zeros_like(self.cov)

        return 0  # in zero'th pass don't request lagged data

    @staticmethod
    def _calc_mean(X, mu, N_mean, traj_len, lag, stride, t, force_eigenvalues_le_one):
        if force_eigenvalues_le_one:
            # MSM-like counting
            if traj_len - lag > 0:
                # find the "tails" of the trajectory relative to the current chunk
                Zptau = lag//stride - t  # zero plus tau
                Nmtau = traj_len - t - lag//stride  # N minus tau

                # restrict them to valid block indices
                size = X.shape[0]
                Zptau = min(max(Zptau, 0), size)
                Nmtau = min(max(Nmtau, 0), size)

                # find start and end of double-counting region
                start2 = min(Zptau, Nmtau)
                end2 = max(Zptau, Nmtau)

                # update mean
                mu += np.sum(X[0:start2, :], axis=0, dtype=np.float64)
                N_mean += start2

                if Nmtau > Zptau: # only if trajectory length > 2*tau, there is double-counting
                    mu += 2.0 * np.sum(X[start2:end2, :], axis=0, dtype=np.float64)
                    N_mean += 2.0 * (end2 - start2)

                mu += np.sum(X[end2:, :], axis=0, dtype=np.float64)
                N_mean += (size - end2)
        else:
            # traditional counting
            mu += np.sum(X, axis=0, dtype=np.float64)
            N_mean += np.shape(X)[0]

        return N_mean

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        r"""
        Chunk-based parameterization of TICA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance and time-lagged covariance
        matrices are estimated. Finally, the generalized eigenvalue problem is solved to determine
        the independent components.

        :param X:
            coordinates. axis 0: time, axes 1-..: coordinates
        :param itraj:
            index of the current trajectory
        :param t:
            time index of first frame within trajectory
        :param first_chunk:
            boolean. True if this is the first chunk globally.
        :param last_chunk_in_traj:
            boolean. True if this is the last chunk within the trajectory.
        :param last_chunk:
            boolean. True if this is the last chunk globally.
        :param ipass:
            number of pass through data
        :param Y:
            time-lagged data (if available)
        :return:
        """
        if ipass == 0:

            # if we have a user-given mean, skip ipass 0 now:
            if self._given_mean:
                if self._force_eigenvalues_le_one:
                    self._logger.warning("Constraint of eigenvalues <= 1 is active,"
                                         "so the mean also depends on the lag time!")
                raise SkipPassException(self._lag, stride)

            traj_len = self.trajectory_length(itraj, stride=1)
            self._N_mean = self._calc_mean(X, self.mu, self._N_mean, traj_len,
                                           self._lag, stride, t, self._force_eigenvalues_le_one)

            # counting chunks and log of eta
            self._progress_update(1, stage=0)

            if last_chunk:
                self.mu /= self._N_mean

                # now we request real lagged data, since we are finished
                # with first pass
                return False, self._lag

        elif ipass == 1:

            if self.trajectory_length(itraj, stride=1) - self._lag > 0:

                self._N_cov_tau += 2.0 * np.shape(Y)[0]
                # _N_cov_tau is muliplied by 2, because we later symmetrize
                # cov_tau, so we are actually using twice the number of samples
                # for every element.
                X_meanfree = X - self.mu
                Y_meanfree = Y - self.mu
                # update the time-lagged covariance matrix
                end = min(X_meanfree.shape[0], Y_meanfree.shape[0])
                self.cov_tau += 2.0 * np.dot(X_meanfree[0:end].T,
                                             Y_meanfree[0:end])

                # update the instantaneous covariance matrix
                if self._force_eigenvalues_le_one:
                    # MSM-like counting
                    Zptau = self._lag//stride-t  # zero plus tau
                    Nmtau = self.trajectory_length(itraj, stride=stride)-t-self._lag//stride  # N minus tau

                    # restrict to valid block indices
                    size = X_meanfree.shape[0]
                    Zptau = min(max(Zptau, 0), size)
                    Nmtau = min(max(Nmtau, 0), size)

                    # update covariance matrix
                    start2 = min(Zptau, Nmtau)
                    end2 = max(Zptau, Nmtau)
                    self.cov += np.dot(X_meanfree[0:start2, :].T,
                                       X_meanfree[0:start2, :])
                    self._N_cov += start2

                    if Nmtau > Zptau:
                        self.cov += 2.0 * np.dot(X_meanfree[start2:end2, :].T,
                                                 X_meanfree[start2:end2, :])
                        self._N_cov += 2.0 * (end2 - start2)

                    self.cov += np.dot(X_meanfree[end2:, :].T,
                                       X_meanfree[end2:, :])
                    self._N_cov += (size - end2)
                else:
                    # traditional counting
                    self.cov += 2.0 * np.dot(X_meanfree.T, X_meanfree)
                    self._N_cov += 2.0 * np.shape(X)[0]

                self._progress_update(1, stage=1)

            else:
                self._skipped_trajs.append(itraj)

            if last_chunk:
                return True  # finished!

        return False  # not finished yet.

    def _param_finish(self):
        if self._force_eigenvalues_le_one:
            assert self._N_mean == self._N_cov, 'inconsistency in C(0) and mu'
            assert self._N_cov == self._N_cov_tau, 'inconsistency in C(0) and C(tau)'

        self._logger.debug("Estimating two covariance matrices"
                           " with dimension (%i, %i)" % (len(self.cov), len(self.cov)))

        # symmetrize covariance matrices
        self.cov = self.cov + self.cov.T
        self.cov *= 0.5

        self.cov_tau = self.cov_tau + self.cov_tau.T
        self.cov_tau *= 0.5

        # norm
        self.cov /= self._N_cov - 2
        self.cov_tau /= self._N_cov_tau - 2

        # diagonalize with low rank approximation
        self._logger.debug("diagonalize Cov and Cov_tau.")
        self._eigenvalues, self._eigenvectors = \
            eig_corr(self.cov, self.cov_tau, self._epsilon)
        self._logger.debug("finished diagonalisation.")

        # compute cumulative variance
        self._cumvar = np.cumsum(self._eigenvalues ** 2)
        self._cumvar /= self._cumvar[-1]

        if len(self._skipped_trajs) >= 1:
            self._skipped_trajs = np.asarray(self._skipped_trajs)
            self._logger.warn("Had to skip %u trajectories for being too short. "
                              "Their indexes are in self._skipped_trajs."%len(self._skipped_trajs))

    def _transform_array(self, X):
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        # TODO: consider writing an extension to avoid temporary Xmeanfree
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self._eigenvectors[:, 0:self.dimension()])
        if self._kinetic_map:  # scale by eigenvalues
            Y *= self._eigenvalues[0:self.dimension()]
        return Y

    @property
    def feature_TIC_correlation(self):
        r"""Instantaneous correlation matrix between input features and TICs

        Denoting the input features as :math:`X_i` and the TICs as :math:`\theta_j`, the instantaneous, linear correlation
        between them can be written as

        .. math::

            \mathbf{Corr}(X_i, \mathbf{\theta}_j) = \frac{1}{\sigma_{X_i}}\sum_l \sigma_{X_iX_l} \mathbf{U}_{li}

        The matrix :math:`\mathbf{U}` is the matrix containing, as column vectors, the eigenvectors of the TICA
        generalized eigenvalue problem .

        Returns
        -------
        feature_TIC_correlation : ndarray(n,m)
            correlation matrix between input features and TICs. There is a row for each feature and a column
            for each TIC.
        """
        feature_sigma = np.sqrt(np.diag(self.cov))
        return np.dot(self.cov, self._eigenvectors[:, : self.dimension()]) / feature_sigma[:, np.newaxis]

    @property
    def timescales(self):
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :py:obj:`lag` of the TICA object and :math:`\lambda_i` is the `i`-th
        :py:obj:`eigenvalue <eigenvalues>` of the TICA object.

        Returns
        -------
        timescales: 1D np.array
            numpy array with the implied timescales. In principle, one should expect as many timescales as
            input coordinates were available. However, less eigenvalues will be returned if the TICA matrices
            were not full rank or :py:obj:`var_cutoff` was parsed
        """
        if self._parametrized:
            return -self.lag/np.log(np.abs(self.eigenvalues))
        else:
            self._logger.info("TICA not yet parametrized")

    @property
    def eigenvalues(self):
        r"""Eigenvalues of the TICA problem (usually denoted :math:`\lambda`

        Returns
        -------
        eigenvalues: 1D np.array
        """

        if self._parametrized:
            return(self._eigenvalues)
        else:
            self._logger.info("TICA not yet parametrized")

    @property
    def eigenvectors(self):
        r"""Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """

        if self._parametrized:
            return(self._eigenvectors)
        else:
            self._logger.info("TICA not yet parametrized")

    @property
    def cumvar(self):
        r"""Cumulative sum of the the TICA eigenvalues

        Returns
        -------
        cumvar: 1D np.array
        """

        if self._parametrized:
            return(self._cumvar)
        else:
            self._logger.info("TICA not yet parametrized")
