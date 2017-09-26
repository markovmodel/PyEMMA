# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
@author: paul, marscher, wu, noe
'''

from __future__ import absolute_import

import numpy as np
from decorator import decorator
from pyemma._base.model import Model
from pyemma.util.annotators import fix_docs
from pyemma.util.types import ensure_ndarray_or_None, ensure_ndarray
from pyemma._ext.variational.solvers.direct import spd_inv_sqrt
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer
import warnings

__all__ = ['VAMP']


class VAMPModel(Model):
    # TODO: remove dummy when bugfix from Martin is committed
    def set_model_params(self, dummy, mean_0, mean_t, C00, Ctt, C0t):
        self.mean_0 = mean_0
        self.mean_t = mean_t
        self.C00 = C00
        self.Ctt = Ctt
        self.C0t = C0t


@decorator
def _lazy_estimation(func, *args, **kw):
    assert isinstance(args[0], VAMP)
    tica_obj = args[0]
    if not tica_obj._estimated:
        tica_obj._diagonalize()
    return func(*args, **kw)


@fix_docs
class VAMP(StreamingEstimationTransformer):
    r"""Variational approach for Markov processes (VAMP)"""

    def describe(self):
        return "[VAMP, lag = %i; max. output dim. = %s]" % (self._lag, str(self.dim))

    def __init__(self, lag, dim=None, scaling=None, right=True, epsilon=1e-6,
                 stride=1, skip=0, ncov_max=float('inf')):
        r""" Variational approach for Markov processes (VAMP) [1]_.

        Parameters
        ----------
        lag : int
            lag time
        dim : float or int
            Number of dimensions to keep:
            * if dim is not set all available ranks are kept:
                n_components == min(n_samples, n_features)
            * if dim is an integer >= 1, this number specifies the number
              of dimensions to keep. By default this will use the kinetic
              variance.
            * if dim is a float with ``0 < dim < 1``, select the number
              of dimensions such that the amount of kinetic variance
              that needs to be explained is greater than the percentage
              specified by dim.
        scaling : None or string
            Scaling to be applied to the VAMP modes upon transformation
            * None: no scaling will be applied, variance along the mode is 1
            * 'kinetic map' or 'km': modes are scaled by singular value
        right : boolean
            Whether to compute the right singular functions.
            If right==True, get_output() will return the right singular
            functions. Otherwise, get_output() will return the left singular
            functions.
            Beware that only frames[tau:, :] of each trajectory returned
            by get_output() contain valid values of the right singular
            functions. Conversely, only frames[0:-tau, :] of each
            trajectory returned by get_output() contain valid values of
            the left singular functions. The remaining frames might
            possibly be interpreted as some extrapolation.
        epsilon : float
            singular value cutoff. Singular values of C0 with norms <= epsilon
            will be cut off. The remaining number of singular values define
            the size of the output.
        stride: int, optional, default = 1
            Use only every stride-th time step. By default, every time step is used.
        skip : int, default=0
            skip the first initial n frames per trajectory.


        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        """
        StreamingEstimationTransformer.__init__(self)

        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=True, remove_data_mean=True, reversible=False,
                                       lag=lag, bessel=False, stride=stride, skip=skip, weights=None, ncov_max=ncov_max)

        # empty dummy model instance
        self._model = VAMPModel()
        self.set_params(lag=lag, dim=dim, scaling=scaling, right=right,
                        epsilon=epsilon, stride=stride, skip=skip, ncov_max=ncov_max)

    def _estimate(self, iterable, **kw):
        indim = iterable.dimension()

        if isinstance(self.dim, int):
            if not self.dim <= indim:
                raise RuntimeError("requested more output dimensions (%i) than dimension"
                                   " of input data (%i)" % (self.dim, indim))

        if self._logger_is_active(self._loglevel_DEBUG):
            self._logger.debug("Running VAMP with tau=%i; Estimating two covariance matrices"
                               " with dimension (%i, %i)" % (self._lag, indim, indim))

        self._covar.estimate(iterable, **kw)
        self._model.update_model_params(mean_0=self._covar.mean,
                                        mean_t=self._covar.mean_tau,
                                        C00=self._covar.C00_,
                                        C0t=self._covar.C0t_,
                                        Ctt=self._covar.Ctt_)
        self._diagonalize()

        return self._model

    def partial_fit(self, X):
        """ incrementally update the covariances and mean.

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.

        Notes
        -----
        The projection matrix is first being calculated upon its first access.
        """
        from pyemma.coordinates import source
        iterable = source(X)

        if isinstance(self.dim, int):
            indim = iterable.dimension()
            if not self.dim <= indim:
                raise RuntimeError("requested more output dimensions (%i) than dimension"
                                   " of input data (%i)" % (self.dim, indim))

        self._covar.partial_fit(iterable)
        self._model.update_model_params(mean_0=self._covar.mean, # TODO: inefficient, fixme
                                        mean_t=self._covar.mean_tau,
                                        C00=self._covar.C00_,
                                        C0t=self._covar.C0t_,
                                        Ctt=self._covar.Ctt_)

        #self._used_data = self._covar._used_data
        self._estimated = False

        return self

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self._logger.debug("diagonalize covariance matrices")

        mean_0 = self._covar.mean
        mean_t = self._covar.mean_tau
        L0 = spd_inv_sqrt(self._covar.C00_)
        Lt = spd_inv_sqrt(self._covar.Ctt_)
        A = L0.T.dot(self._covar.C0t_).dot(Lt)

        Uprime, s, Vprimeh = np.linalg.svd(A, compute_uv=True)

        # compute cumulative variance
        cumvar = np.cumsum(s ** 2)
        cumvar /= cumvar[-1]

        self._L0 = L0
        self._Lt = Lt
        self._model.update_model_params(cumvar=cumvar, singular_values=s, mean_0=mean_0, mean_t=mean_t)

        m = self.dimension(_estimating=True)

        U = L0.dot(Uprime[:, :m]) # U in the paper singular_vectors_left
        V = Lt.dot(Vprimeh[:m, :].T) # V in the paper singular_vectors_right

        # normalize vectors
        #scale_left = np.diag(singular_vectors_left.T.dot(self._model.C00).dot(singular_vectors_left))
        #scale_right = np.diag(singular_vectors_right.T.dot(self._model.Ctt).dot(singular_vectors_right))
        #singular_vectors_left *= scale_left[np.newaxis, :]**-0.5
        #singular_vectors_right *= scale_right[np.newaxis, :]**-0.5

        # scale vectors
        if self.scaling is None:
            pass
        elif self.scaling in ['km', 'kinetic map']:
            U *= self.singular_values[np.newaxis, :] ## TODO: check left/right, ask Hao
            V *= self.singular_values[np.newaxis, :] ## TODO: check left/right, ask Hao
        else:
            raise ValueError('unexpected value (%s) of "scaling"' % self.scaling)

        self._logger.debug("finished diagonalisation.")

        self._model.update_model_params(U=U, V=V)

        self._estimated = True


    def dimension(self, _estimating=False):
        """ output dimension """
        if self.dim is None or self.dim == 1.0:
            if self._estimated or _estimating:
                return np.count_nonzero(self._model.singular_values > self.epsilon)
            else:
                warnings.warn(
                    RuntimeWarning('Requested dimension, but the dimension depends on the singular values and the '
                                   'transformer has not yet been estimated. Result is only an approximation.'))
                return self.data_producer.dimension()
        if isinstance(self.dim, float):
            if self._estimated or _estimating:
                return np.count_nonzero(self._model.cumvar >= self.dim)
            else:
                raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                                   'transformer has not yet been estimated. Call estimate() before.')
        else:
            if self._estimated or _estimating:
                return min(np.min(np.count_nonzero(self._model.singular_values > self.epsilon)), self.dim)
            else:
                warnings.warn(
                    RuntimeWarning('Requested dimension, but the dimension depends on the singular values and the '
                                   'transformer has not yet been estimated. Result is only an approximation.'))
                return self.dim

    def _transform_array(self, X):
        r"""Projects the data onto the dominant singular functions.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        # TODO: in principle get_output should not return data for *all* frames!
        # TODO: implement our own iterators? This would also include random access to be complete...
        if self.right:
            X_meanfree = X - self._model.mean_t
            Y = np.dot(X_meanfree, self._model.V[:, 0:self.dimension()])
        else:
            X_meanfree = X - self._model.mean_0
            Y = np.dot(X_meanfree, self._model.U[:, 0:self.dimension()])

        return Y.astype(self.output_type())

    def output_type(self):
        return StreamingEstimationTransformer.output_type(self)

    @property
    @_lazy_estimation
    def singular_values(self):
        r"""Singular values of VAMP (usually denoted :math:`\sigma`)

        Returns
        -------
        singular values: 1-D np.array
        """
        return self._model.singular_values

    @property
    @_lazy_estimation
    def singular_vectors_right(self):
        r"""Right singular vectors V of the VAMP problem, columnwise

        Returns
        -------
        eigenvectors: 2-D ndarray
        Coefficients that express the right singular functions in the
        basis of mean-free input features.
        """
        return self._model.V

    @property
    @_lazy_estimation
    def singular_vectors_left(self):
        r"""Left singular vectors U of the VAMP problem, columnwise

        Returns
        -------
        eigenvectors: 2-D ndarray
        Coefficients that express the left singular functions in the
        basis of mean-free input features.
        """
        return self._model.U

    @property
    @_lazy_estimation
    def cumvar(self):
        r"""Cumulative sum of the squared and normalized VAMP singular values

        Returns
        -------
        cumvar: 1D np.array
        """
        return self._model.cumvar


    def expectation(self, statistics, observables, lag_multiple=1, statistics_mean_free=False, observables_mean_free=False):
        r"""Compute future expectation of observable or covariance using the approximated Koopman operator.

        TODO: this requires some discussion

        TODO: add equations

        Parameters
        ----------
        statistics : np.ndarray((input_dimension, n_statistics)), optional
            Coefficients that express one or multiple statistics in
            the basis of the input features.
            This parameter can be None. In that case, this method
            returns the future expectation value of the observable(s).

        observables : np.ndarray((input_dimension, n_observables))
            Coefficients that express one or multiple observables in
            the basis of the input features.

        lag_multiple : int
            If > 1, extrapolate to a multiple of the estimator's lag
            time by assuming Markovianity of the approximated Koopman
            operator.

        statistics_mean_free : bool, default=False
            If true, coefficients in statistics refer to the input
            features with feature means removed.
            If false, coefficients in statistics refer to the
            unmodified input features.

        observables_mean_free : bool, default=False
            If true, coefficients in observables refer to the input
            features with feature means removed.
            If false, coefficients in observables refer to the
            unmodified input features.
        """
        import sys

        S = np.diag(np.concatenate(([1.0], self.singular_values[0:self.dimension()])))
        V = self.singular_vectors_right[:, 0:self.dimension()]
        U = self.singular_vectors_left[:, 0:self.dimension()]
        m_0 = self.model.mean_0
        m_t = self.model.mean_t

        dim = self.dimension()

        assert lag_multiple >= 1, 'lag_multiple = 0 not implemented'

        if lag_multiple == 1:
            P = S
        else:
            p = np.zeros((dim + 1, dim + 1))
            p[0, 0] = 1.0
            p[1:, 0] = U.T.dot(m_t - m_0)
            p[1:, 1:] = U.T.dot(self.model.Ctt).dot(V)
            P = np.linalg.matrix_power(S.dot(p), lag_multiple - 1).dot(S)

        Q = np.zeros((observables.shape[1], dim + 1))
        if not observables_mean_free:
            Q[:, 0] = observables.T.dot(m_t)
        Q[:, 1:] = observables.T.dot(self.model.Ctt).dot(V)

        if statistics is not None:
            # compute covariance
            R = np.zeros((statistics.shape[1], dim + 1))
            if not statistics_mean_free:
                R[:, 0] = statistics.T.dot(m_0)
            R[:, 1:] = statistics.T.dot(self.model.C00).dot(U)

        if statistics is not None:
            # compute lagged covariance
            return Q.dot(P).dot(R.T)
        else:
            # compute future expectation
            return Q.dot(P)[:, 0]


    def cktest(self, n_observables=None, observables='psi', statistics='phi', mlags=10):
        # TODO: make better API, discuss
        #from pyemma._ext.sklearn.base import clone as clone_estimator

        if n_observables is not None:
            if n_observables > self.dimension():
                warnings.warn('Selected singular functions as observables but dimension '
                              'is lower than requested number of observables.')
                n_observables = self.dimension()
        else:
            n_observables = self.dimension()

        if isinstance(observables, str) and observables == 'psi':
            observables = self.singular_vectors_right[:, 0:n_observables]
            observables_mean_free = True
        else:
            ensure_ndarray(observables, ndim=2)
            observables_mean_free = False

        if isinstance(statistics, str) and statistics == 'phi':
            statistics = self.singular_vectors_left[:, 0:n_observables]
            statistics_mean_free = True
        else:
            ensure_ndarray_or_None(statistics, ndim=2)
            statistics_mean_free = False

        est_1 = self.expectation(statistics, observables, lag_multiple=1, statistics_mean_free=statistics_mean_free,
                                 observables_mean_free=observables_mean_free)
        estimates = [est_1]
        predictions = [est_1]
        for m in np.arange(2, mlags+1):
            #copy = clone_estimator(self) # TODO: why doesn't this work?
            copy = VAMP(lag=self.lag*m, dim=self.dim, scaling=self.scaling, right=self.right,
                        epsilon=self.epsilon, stride=self.stride, skip=self.skip,
                        ncov_max=self.ncov_max)
            #copy.lag = self.lag*m
            estimates.append(
                copy.estimate(self.data_producer).expectation(statistics, observables, lag_multiple=1,
                                                              statistics_mean_free=statistics_mean_free,
                                                              observables_mean_free=observables_mean_free))
            predictions.append(
                self.expectation(statistics, observables, lag_multiple=m, statistics_mean_free=statistics_mean_free,
                                 observables_mean_free=observables_mean_free))
        # TODO: create some fancy object to store results
        return predictions, estimates
