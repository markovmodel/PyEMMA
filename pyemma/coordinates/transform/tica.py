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
'''
Created on 19.01.2015

@author: marscher
'''
from .transformer import Transformer

from pyemma.util.progressbar import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar
from pyemma.util.linalg import eig_corr
from pyemma.util.annotators import doc_inherit

import numpy as np

__all__ = ['TICA']


class TICA(Transformer):

    def __init__(self, lag, dim=-1, var_cutoff=1.0, kinetic_map=False, epsilon=1e-6,
                 force_eigenvalues_le_one=False):
        r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
        tau : int
            lag time
        dim : int, optional, default -1
            Maximum number of significant independent components to use to reduce dimension of input data. -1 means
            all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.
        var_cutoff : float in the range [0,1], optional, default 1
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim
        kinetic_map : bool, optional, default False
            Eigenvectors will be scaled by eigenvalues. As a result, Euclidean distances in the transformed data
            approximate kinetic distances [4]_. This is a good choice when the data is further processed by clustering.
        epsilon : float
            eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
        force_eigenvalues_le_one : boolean
            Compute covariance matrix and time-lagged covariance matrix such
            that the generalized eigenvalues are always guaranteed to be <= 1.

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
            (in preparation).

        """
        super(TICA, self).__init__()

        # store lag time to set it appropriately in second pass of parametrize
        self._lag = lag
        self._dim = dim
        self._var_cutoff = var_cutoff
        if dim != -1 and var_cutoff < 1.0:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')
        self._kinetic_map = kinetic_map
        self._epsilon = epsilon
        self._force_eigenvalues_le_one = force_eigenvalues_le_one

        # covariances
        self.cov = None
        self.cov_tau = None
        # mean
        self.mu = None
        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0
        self.eigenvalues = None
        self.eigenvectors = None
        self.cumvar = None

        self._custom_param_progress_handling = True
        self._progress_mean = None
        self._progress_cov = None

        # skipped trajectories
        self._skipped_trajs = []
    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        self._parametrized = False
        self._lag = new_tau

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
        if self._dim != -1:  # fixed parametrization
            return self._dim
        elif self._parametrized:  # parametrization finished. Dimension is known
            dim = len(self.eigenvalues)
            if self._var_cutoff < 1.0:  # if subspace_variance, reduce the output dimension if needed
                dim = min(dim, np.searchsorted(self.cumvar, self._var_cutoff)+1)
            return dim
        elif self._var_cutoff == 1.0:  # We only know that all dimensions are wanted, so return input dim
            return self.data_producer.dimension()
        else:  # We know nothing. Give up
            raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                               'transformer has not yet been parametrized. Call parametrize() before.')

    @property
    def mean(self):
        """ mean of input features """
        return self.mu

    def _param_init(self):
        indim = self.data_producer.dimension()
        assert indim > 0, "zero dimension from data producer"
        assert self._dim <= indim, ("requested more output dimensions (%i) than dimension"
                                    " of input data (%i)" % (self._dim, indim))

        self._N_mean = 0
        self._N_cov = 0
        self._N_cov_tau = 0
        # create mean array and covariance matrices
        self.mu = np.zeros(indim)

        self.cov = np.zeros((indim, indim))
        self.cov_tau = np.zeros_like(self.cov)

        self._logger.debug("Running TICA with tau=%i; Estimating two covariance matrices"
                           " with dimension (%i, %i)" % (self._lag, indim, indim))

        # amount of chunks
        denom = self._n_chunks(self._param_with_stride)
        self._progress_mean = ProgressBar(denom, description="calculate mean")
        self._progress_cov = ProgressBar(denom, description="calculate covariances")

        return 0  # in zero'th pass don't request lagged data

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

            if self._force_eigenvalues_le_one:
                # MSM-like counting
                if self.trajectory_length(itraj, stride=stride) > self._lag:
                    # find the "tails" of the trajectory relative to the current chunk
                    Zptau = self._lag-t  # zero plus tau
                    Nmtau = self.trajectory_length(itraj, stride=stride)-t-self._lag  # N minus tau

                    # restrict them to valid block indices
                    size = X.shape[0]
                    Zptau = min(max(Zptau, 0), size)
                    Nmtau = min(max(Nmtau, 0), size)

                    # find start and end of double-counting region
                    start2 = min(Zptau, Nmtau)
                    end2 = max(Zptau, Nmtau)

                    # update mean
                    self.mu += np.sum(X[0:start2, :], axis=0, dtype=np.float64)
                    self._N_mean += start2

                    if Nmtau > Zptau: # only if trajectory length > 2*tau, there is double-counting
                        self.mu += 2.0 * np.sum(X[start2:end2, :], axis=0, dtype=np.float64)
                        self._N_mean += 2.0 * (end2 - start2)

                    self.mu += np.sum(X[end2:, :], axis=0, dtype=np.float64)
                    self._N_mean += (size - end2)
            else:
                # traditional counting
                self.mu += np.sum(X, axis=0, dtype=np.float64)
                self._N_mean += np.shape(X)[0]

            # counting chunks and log of eta
            self._progress_mean.numerator += 1
            show_progressbar(self._progress_mean)

            if last_chunk:
                self.mu /= self._N_mean

                # now we request real lagged data, since we are finished
                # with first pass
                return False, self._lag

        elif ipass == 1:

            if self.trajectory_length(itraj, stride=stride) > self._lag:
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
                    Zptau = self._lag-t  # zero plus tau
                    Nmtau = self.trajectory_length(itraj, stride=stride)-t-self._lag  # N minus tau

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

                self._progress_cov.numerator += 1
                show_progressbar(self._progress_cov)

            else:
                self._skipped_trajs.append(itraj)

            if last_chunk:
                return True  # finished!

        return False  # not finished yet.

    def _param_finish(self):
        if self._force_eigenvalues_le_one:
            assert self._N_mean == self._N_cov, 'inconsistency in C(0) and mu'
            assert self._N_cov == self._N_cov_tau, 'inconsistency in C(0) and C(tau)'

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
        self.eigenvalues, self.eigenvectors = \
            eig_corr(self.cov, self.cov_tau, self._epsilon)
        self._logger.debug("finished diagonalisation.")

        # compute cumulative variance
        self.cumvar = np.cumsum(self.eigenvalues ** 2)
        self.cumvar /= self.cumvar[-1]


        if len(self._skipped_trajs) >= 1:
            self._skipped_trajs = np.asarray(self._skipped_trajs)
            self._logger.warn("Had to skip %u trajectories for being too short. "
                              "Their indexes are in self._skipped_trajs."%len(self._skipped_trajs))

    def _map_array(self, X):
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
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self.dimension()])
        if self._kinetic_map:  # scale by eigenvalues
            Y *= self.eigenvalues[0:self.dimension()]
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
        return np.dot(self.cov, self.eigenvectors[:, : self.dimension()]) / feature_sigma[:, np.newaxis]
