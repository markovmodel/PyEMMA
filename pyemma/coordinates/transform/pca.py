
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
import numpy as np

from pyemma.util.annotators import doc_inherit
from pyemma.coordinates.transform.transformer import SkipPassException, Transformer
from pyemma.util import types
from pyemma.util.reflection import get_default_args

__all__ = ['PCA']
__author__ = 'noe'


class PCA(Transformer):
    r""" Principal component analysis."""

    def __init__(self, dim=-1, var_cutoff=0.95, mean=None):
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

        var_cutoff : float in the range [0,1], optional, default 0.95
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim

        mean : ndarray, optional, default None
            Optionally pass pre-calculated means to avoid their re-computation.
            The shape has to match the input dimension.

        """
        super(PCA, self).__init__()
        self._dim = dim
        self._var_cutoff = var_cutoff
        default_var_cutoff = get_default_args(self.__init__)['var_cutoff']
        if dim != -1 and var_cutoff != default_var_cutoff:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')
        self.Y = None
        self._N_mean = 0
        self._N_cov = 0

        self.mu = mean

        # set up result variables
        self.eigenvalues = None
        self.eigenvectors = None
        self.cumvar = None

        # output options
        self._custom_param_progress_handling = True

    @doc_inherit
    def describe(self):
        return "[PCA, output dimension = %i]" % self._dim

    def dimension(self):
        """ output dimension """
        d = None
        if self._dim != -1:  # fixed parametrization
            d = self._dim
        elif self._parametrized:  # parametrization finished. Dimension is known
            dim = len(self.eigenvalues)
            if self._var_cutoff < 1.0:  # if subspace_variance, reduce the output dimension if needed
                dim = min(dim, np.searchsorted(self.cumvar, self._var_cutoff)+1)
            d = dim
        elif self._var_cutoff == 1.0:  # We only know that all dimensions are wanted, so return input dim
            d = self.data_producer.dimension()
        else:  # We know nothing. Give up
            raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                               'transformer has not yet been parametrized. Call parametrize() before.')
        return d

    @property
    def mean(self):
        return self.mu

    @property
    def covariance_matrix(self):
        return self.cov

    def _param_init(self):
        self._N_mean = 0
        self._N_cov = 0
        # create mean array and covariance matrix
        indim = self.data_producer.dimension()
        self._logger.info("Running PCA on %i dimensional input" % indim)
        assert indim > 0, "Incoming data of PCA has 0 dimension!"

        if self.mu is not None:
            self.mu = types.ensure_ndarray(self.mu, shape=(indim,))
            self._given_mean = True
        else:
            self.mu = np.zeros(indim)
            self._given_mean = False

        self.cov = np.zeros((indim, indim))

        # amount of chunks
        denom = self._n_chunks(self._param_with_stride)
        self._progress_register(denom, description="calculate mean", stage=0)
        self._progress_register(denom, description="calculate covariances", stage=1)

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
                if self._given_mean:
                    raise SkipPassException(next_pass_stride=stride)

            self.mu += np.sum(X, axis=0)
            self._N_mean += np.shape(X)[0]

            # counting chunks and log of eta
            self._progress_update(1, 0)

            if last_chunk:
                self.mu /= self._N_mean

        # pass 2: covariances
        if ipass == 1:
            if t == 0:
                self._logger.debug("start calculate covariance for traj nr %i" % itraj)
            Xm = X - self.mu
            self.cov += np.dot(Xm.T, Xm)
            self._N_cov += np.shape(X)[0]

            self._progress_update(1, stage=1)

            if last_chunk:
                self.cov /= self._N_cov - 1
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

    def _transform_array(self, X):
        r"""
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        # TODO: consider writing an extension to avoid temporary Xmeanfree
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self.dimension()])
        return Y
