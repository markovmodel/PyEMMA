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

__author__ = 'noe'

import numpy as np

from .transformer import Transformer

from pyemma.util.annotators import doc_inherit
from pyemma.util.progressbar import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar

__all__ = ['PCA']


class PCA(Transformer):

    r"""Principal component analysis.

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
    output_dimension : int
        number of principal components to project onto

    """

    def __init__(self, output_dimension):
        super(PCA, self).__init__()
        self._output_dimension = output_dimension
        self._dot_prod_tmp = None
        self.Y = None

        self._progress_mean = None
        self._progress_cov = None

    @doc_inherit
    def describe(self):
        return "[PCA, output dimension = %i]" % self._output_dimension

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self._output_dimension

    @doc_inherit
    def _get_constant_memory(self):
        """Returns the constant memory requirements, in bytes."""
        # memory for mu, C, v, R
        dim = self.data_producer.dimension()

        cov_elements = dim ** 2
        mu_elements = dim

        v_elements = dim
        R_elements = cov_elements

        return 8 * (cov_elements + mu_elements + v_elements + R_elements)

    @doc_inherit
    def _get_memory_per_frame(self):
        # memory for temporaries
        dim = self.data_producer.dimension()

        x_meanfree_elements = self.chunksize * dim

        dot_prod_elements = dim

        return 8 * (x_meanfree_elements + dot_prod_elements)

    @property
    def mean(self):
        return self.mu

    @property
    def covariance_matrix(self):
        return self.cov

    @doc_inherit
    def _param_init(self):
        self.N = 0
        # create mean array and covariance matrix
        dim = self.data_producer.dimension()
        self._logger.info("Running PCA on %i dimensional input" % dim)
        assert dim > 0, "Incoming data of PCA has 0 dimension!"
        self.mu = np.zeros(dim)
        self.cov = np.zeros((dim, dim))

        # amount of chunks
        denom = self._n_chunks(self._param_with_stride)
        self._progress_mean = ProgressBar(denom, description="calculate mean")
        self._progress_cov = ProgressBar(denom, description="calculate covariances")

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        """
        Chunk-based parametrization of PCA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance matrix is estimated.
        Finally, the eigenvalue problem is solved to determine the principal compoennts.

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
            self.N += np.shape(X)[0]

            # counting chunks and log of eta
            self._progress_mean.numerator += 1
            show_progressbar(self._progress_mean)

            if last_chunk:
                self.mu /= self.N

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
                self.cov /= self.N - 1
                self._logger.debug("finished")
                return True  # finished!

        # by default, continue
        return False

    @doc_inherit
    def _param_finish(self):
        (v, R) = np.linalg.eigh(self.cov)
        # sort
        I = np.argsort(v)[::-1]
        self.eigenvalues = v[I]
        self.eigenvectors = R[:, I]

    def _map_array(self, X):
        """
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self._output_dimension])
        return Y