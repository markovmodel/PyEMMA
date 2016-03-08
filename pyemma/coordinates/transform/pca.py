
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

import math

from pyemma._base.model import Model
from pyemma._base.progress.reporter import ProgressReporter
from pyemma.coordinates.transform.transformer import StreamingTransformer
from pyemma.util.annotators import doc_inherit
from pyemma.util.reflection import get_default_args

from pyemma.coordinates.estimators.covar.running_moments import running_covar
import numpy as np
from decorator import decorator


__all__ = ['PCA']
__author__ = 'noe'


@decorator
def _lazy_estimation(func, *args, **kw):
    assert isinstance(args[0], PCA)
    tica_obj = args[0]
    if not tica_obj._estimated:
        tica_obj._diagonalize()
    return func(*args, **kw)


class PCAModel(Model):
    # todo: do we really want this?
    pass


class PCA(StreamingTransformer, ProgressReporter):
    r""" Principal component analysis."""

    def __init__(self, dim=-1, var_cutoff=0.95, mean=None, stride=1):
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
        default_var_cutoff = get_default_args(self.__init__)['var_cutoff']
        if dim != -1 and var_cutoff != default_var_cutoff:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')

        self._model = PCAModel()
        self.set_params(dim=dim, var_cutoff=var_cutoff, mean=mean)
        self._model = PCAModel()

    @doc_inherit
    def describe(self):
        return "[PCA, output dimension = %i]" % self.dim

    def dimension(self):
        """ output dimension """
        if self.dim != -1:  # fixed parametrization
            d = self.dim
        elif self._estimated:  # estimation finished. Dimension is known
            dim = len(self._model.eigenvalues)
            if self.var_cutoff < 1.0:  # if subspace_variance, reduce the output dimension if needed
                dim = min(dim, np.searchsorted(self.cumvar, self.var_cutoff)+1)
            d = dim
        elif self.var_cutoff == 1.0:  # We only know that all dimensions are wanted, so return input dim
            d = self.data_producer.dimension()
        else:  # We know nothing. Give up
            raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                               'transformer has not yet been parametrized. Call parametrize() before.')
        return d

    @property
    @_lazy_estimation
    def cumvar(self):
        return self._model.cumvar

    @cumvar.setter
    def cumvar(self, value):
        self._model.cumvar = value

    @property
    @_lazy_estimation
    def eigenvalues(self):
        return self._model.eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value):
        self._model.eigenvalues = value

    @property
    @_lazy_estimation
    def eigenvectors(self):
        return self._model.eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, value):
        self._model.eigenvectors = value

    @property
    def mean(self):
        return self._model.mean

    @mean.setter
    def mean(self, value):
        self._model.mean = value

    def partial_fit(self, X):
        from pyemma.coordinates import source
        iterable = source(X)

        self._estimate(iterable, partial=True)
        self._estimated = False

        return self

    def _init_covar(self, partial_fit, n_chunks):
        nsave = int(max(math.log(n_chunks, 2), 2))
        # in case we do a one shot estimation, we want to re-initialize running_covar
        if not hasattr(self, '_covar') or not partial_fit:
            self._logger.debug("using %s moments for %i chunks" % (nsave, n_chunks))
            self._covar = running_covar(xx=True, xy=False, yy=False,
                                        remove_mean=True, symmetrize=False,
                                        nsave=nsave)
        else:
            # check storage size vs. n_chunks of the new iterator
            old_nsave = self._covar.storage_XX.nsave
            if old_nsave < nsave or old_nsave > nsave:
                self.logger.info("adopting storage size")
                self._covar.storage_XX.nsave = nsave
                self._covar.storage_XY.nsave = nsave

    def _diagonalize(self):
        (v, R) = np.linalg.eigh(self.cov)
        # sort
        I = np.argsort(v)[::-1]
        eigenvalues = v[I]
        eigenvectors = R[:, I]

        # compute cumulative variance
        cumvar = np.cumsum(eigenvalues)
        cumvar /= cumvar[-1]

        self._model.update_model_params(eigenvalues=eigenvalues,
                                        eigenvectors=eigenvectors,
                                        cumvar=cumvar)

    def _estimate(self, iterable, **kw):
        partial_fit = 'partial' in kw

        with iterable.iterator(return_trajindex=False) as it:
            n_chunks = it._n_chunks
            self._progress_register(n_chunks, "calc mean+cov", 0)
            self._init_covar(partial_fit, n_chunks)

            for chunk in it:
                self._covar.add(chunk)
                self._progress_update(1, 0)

        self.cov = self._covar.cov_XX()
        self.mu = self._covar.mean_X()

        self._model.update_model_params(mean=self._covar.mean_X())
        if not partial_fit:
            self._diagonalize()

        return self._model

    def _transform_array(self, X):
        r"""
        Projects the data onto the dominant principal components.
        :param X: the input data
        :return: the projected data
        """
        X_meanfree = X - self._model.mean
        Y = np.dot(X_meanfree, self._model.eigenvectors[:, 0:self.dimension()])
        return Y
