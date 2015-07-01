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
import numpy as np

from .transformer import Transformer

from pyemma.util.annotators import doc_inherit
from pyemma.util.progressbar import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar

__all__ = ['PCA']
__author__ = 'noe'


class PCA(Transformer):

    def __init__(self, dim=-1, var_cutoff=1.0):
        r""" Principal component analysis.

        Given a sequence of multivariate data :math:`X_t`,
        computes the mean-free covariance matrix.

        .. math:: C = (X - \mu)^T (X - \mu)

        and solves the eigenvalue problem

        .. math:: C r_i = \sigma_i r_i,

        where :math:`r_i` are the principal components and :math:`\sigma_i` are
        their respective variances.

        When used as a dimension reduction method, the input data is projected onto
        the dominant principal components.

        Parameters
        ----------
        dim : int, optional, default -1
            the number of dimensions (independent components) to project onto. A call to the
            :func:`map <pyemma.coordinates.transform.TICA.map>` function reduces the d-dimensional
            input to only dim dimensions such that the data preserves the maximum possible autocorrelation
            amongst dim-dimensional linear projections.
            -1 means all numerically available dimensions will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.

        var_cutoff : float in the range [0,1], optional, default 1
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim


        """
        super(PCA, self).__init__()
        self._dim = dim
        self._var_cutoff = var_cutoff
        if dim != -1 and var_cutoff < 1.0:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')
        self._dot_prod_tmp = None
        self.Y = None
        self._N = 0

        # set up result variables
        self.eigenvalues = None
        self.eigenvectors = None
        self.cumvar = None

        # output options
        self._custom_param_progress_handling = True
        self._progress_mean = None
        self._progress_cov = None

    @doc_inherit
    def describe(self):
        return "[PCA, output dimension = %i]" % self._dim

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
        return self.mu

    @property
    def covariance_matrix(self):
        return self.cov

    def _param_init(self):
        self._N = 0
        # create mean array and covariance matrix
        indim = self.data_producer.dimension()
        self._logger.info("Running PCA on %i dimensional input" % indim)
        assert indim > 0, "Incoming data of PCA has 0 dimension!"
        self.mu = np.zeros(indim)
        self.cov = np.zeros((indim, indim))

        # amount of chunks
        denom = self._n_chunks(self._param_with_stride)
        self._progress_mean = ProgressBar(denom, description="calculate mean")
        self._progress_cov = ProgressBar(denom, description="calculate covariances")

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        r"""
        Chunk-based parametrization of PCA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance matrix is estimated.
        Finally, the eigenvalue problem is solved to determine the principal components.

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
        # pass 1: means
        if ipass == 0:
            if t == 0:
                self._logger.debug("start to calculate mean for traj nr %i" % itraj)
                self._sum_tmp = np.empty(X.shape[1])
            np.sum(X, axis=0, out=self._sum_tmp)
            self.mu += self._sum_tmp
            self._N += np.shape(X)[0]

            # counting chunks and log of eta
            self._progress_mean.numerator += 1
            show_progressbar(self._progress_mean)

            if last_chunk:
                self.mu /= self._N

        # pass 2: covariances
        if ipass == 1:
            if t == 0:
                self._logger.debug("start calculate covariance for traj nr %i" % itraj)
                self._dot_prod_tmp = np.empty_like(self.cov)
            Xm = X - self.mu
            np.dot(Xm.T, Xm, self._dot_prod_tmp)
            self.cov += self._dot_prod_tmp

            self._progress_cov.numerator += 1
            show_progressbar(self._progress_cov)

            if last_chunk:
                self.cov /= self._N - 1
                self._logger.debug("finished")
                return True  # finished!

        # by default, continue
        return False

    def _param_finish(self):
        (v, R) = np.linalg.eigh(self.cov)
        # sort
        I = np.argsort(v)[::-1]
        self.eigenvalues = v[I]
        self.eigenvectors = R[:, I]

        # compute cumulative variance
        self.cumvar = np.cumsum(self.eigenvalues)
        self.cumvar /= self.cumvar[-1]

    def _map_array(self, X):
        r"""
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self._dim])
        return Y