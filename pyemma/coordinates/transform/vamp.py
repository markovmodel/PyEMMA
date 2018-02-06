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

from pyemma._base.model import Model
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.util.annotators import fix_docs
from pyemma.util.types import ensure_ndarray_or_None, ensure_ndarray
from pyemma._ext.variational.solvers.direct import spd_inv_sqrt
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer
from pyemma.msm.estimators.lagged_model_validators import LaggedModelValidator
from pyemma.util.linalg import mdot

import warnings

__all__ = ['VAMP']


class VAMPModel(Model, SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = ('_U', '_V', '_svd_performed')

    def set_model_params(self, mean_0, mean_t, C00, Ctt, C0t, U, V, singular_values, cumvar, dim, epsilon):
        self.mean_0 = mean_0
        self.mean_t = mean_t
        self.C00 = C00
        self.Ctt = Ctt
        self.C0t = C0t
        self._svd_performed = False
        self._U = U
        self._V = V
        self._singular_values = singular_values
        self.cumvar = cumvar
        self.dim = dim
        self.epsilon = epsilon

    @property
    def U(self):
        if not self._svd_performed:
            self._diagonalize()
        return self._U

    @property
    def V(self):
        if not self._svd_performed:
            self._diagonalize()
        return self._V

    @property
    def singular_values(self):
        if not self._svd_performed:
            self._diagonalize()
        return self._singular_values

    @property
    def C00(self):
        return self._C00

    @C00.setter
    def C00(self, val):
        self._svd_performed = False
        self._C00 = val

    @property
    def C0t(self):
        return self._C0t

    @C0t.setter
    def C0t(self, val):
        self._svd_performed = False
        self._C0t = val

    @property
    def Ctt(self):
        return self._Ctt

    @Ctt.setter
    def Ctt(self, val):
        self._svd_performed = False
        self._Ctt = val

    def dimension(self):
        """ output dimension """
        if self.dim is None or (isinstance(self.dim, float) and self.dim == 1.0):
            if hasattr(self, '_singular_values') and self._singular_values is not None:
                return np.count_nonzero(self._singular_values > self.epsilon)
            else:
                raise RuntimeError('Requested dimension, but the dimension depends on the singular values and the '
                                   'transformer has not yet been estimated. Call estimate() before.')
        if isinstance(self.dim, float):
            if hasattr(self, 'cumvar') and self.cumvar is not None:
                return np.count_nonzero(self.cumvar >= self.dim)
            else:
                raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                                   'transformer has not yet been estimated. Call estimate() before.')
        else:
            if hasattr(self, '_singular_values') and self._singular_values is not None:
                return min(np.min(np.count_nonzero(self._singular_values > self.epsilon)), self.dim)
            else:
                warnings.warn(
                    RuntimeWarning('Requested dimension, but the dimension depends on the singular values and the '
                                   'transformer has not yet been estimated. Result is only an approximation.'))
                return self.dim

    def expectation(self, observables, statistics, lag_multiple=1, observables_mean_free=False, statistics_mean_free=False):
        r"""Compute future expectation of observable or covariance using the approximated Koopman operator.

        TODO: add equations

        Parameters
        ----------
        observables : np.ndarray((input_dimension, n_observables))
            Coefficients that express one or multiple observables in
            the basis of the input features.

        statistics : np.ndarray((input_dimension, n_statistics)), optional
            Coefficients that express one or multiple statistics in
            the basis of the input features.
            This parameter can be None. In that case, this method
            returns the future expectation value of the observable(s).

        lag_multiple : int
            If > 1, extrapolate to a multiple of the estimator's lag
            time by assuming Markovianity of the approximated Koopman
            operator.

        observables_mean_free : bool, default=False
            If true, coefficients in observables refer to the input
            features with feature means removed.
            If false, coefficients in observables refer to the
            unmodified input features.
        statistics_mean_free : bool, default=False
            If true, coefficients in statistics refer to the input
            features with feature means removed.
            If false, coefficients in statistics refer to the
            unmodified input features.
        """
        # TODO: implement the case lag_multiple=0

        dim = self.dimension()

        S = np.diag(np.concatenate(([1.0], self.singular_values[0:dim])))
        V = self.V[:, 0:dim]
        U = self.U[:, 0:dim]
        m_0 = self.mean_0
        m_t = self.mean_t

        assert lag_multiple >= 1, 'lag_multiple = 0 not implemented'

        if lag_multiple == 1:
            P = S
        else:
            p = np.zeros((dim + 1, dim + 1))
            p[0, 0] = 1.0
            p[1:, 0] = U.T.dot(m_t - m_0)
            p[1:, 1:] = U.T.dot(self.Ctt).dot(V)
            P = np.linalg.matrix_power(S.dot(p), lag_multiple - 1).dot(S)

        Q = np.zeros((observables.shape[1], dim + 1))
        if not observables_mean_free:
            Q[:, 0] = observables.T.dot(m_t)
        Q[:, 1:] = observables.T.dot(self.Ctt).dot(V)

        if statistics is not None:
            # compute covariance
            R = np.zeros((statistics.shape[1], dim + 1))
            if not statistics_mean_free:
                R[:, 0] = statistics.T.dot(m_0)
            R[:, 1:] = statistics.T.dot(self.C00).dot(U)

        if statistics is not None:
            # compute lagged covariance
            return Q.dot(P).dot(R.T)
            # TODO: discuss whether we want to return this or the transpose
            # TODO: from MSMs one might expect to first index to refer to the statistics, here it is the other way round
        else:
            # compute future expectation
            return Q.dot(P)[:, 0]

    def _diagonalize(self, scaling=None):
        """ performs SVD on covariance matrices and save left, right singular vectors and values in the model.

        Parameters
        ----------
        scaling: str or None

        """

        L0 = spd_inv_sqrt(self.C00)
        Lt = spd_inv_sqrt(self.Ctt)
        A = L0.T.dot(self.C0t).dot(Lt)

        Uprime, s, Vprimeh = np.linalg.svd(A, compute_uv=True)
        self._singular_values = s

        # compute cumulative variance
        cumvar = np.cumsum(s ** 2)
        cumvar /= cumvar[-1]
        self.cumvar = cumvar

        self._L0 = L0
        self._Lt = Lt

        m = self.dimension()

        U = L0.dot(Uprime[:, :m])  # U in the paper singular_vectors_left
        V = Lt.dot(Vprimeh[:m, :].T)  # V in the paper singular_vectors_right

        # scale vectors
        if scaling is None:
            pass
        elif scaling in ['km', 'kinetic map']:
            U *= s[np.newaxis, 0:m]  ## TODO: check left/right, ask Hao
            V *= s[np.newaxis, 0:m]  ## TODO: check left/right, ask Hao
        else:
            raise ValueError('unexpected value (%s) of "scaling"' % scaling)

        self._U = U
        self._V = V
        self._svd_performed = True

    def score(self, test_model=None, score_method='VAMP2'):
        """

        Parameters
        ----------
        test_model
        score_method : str, optional, default='VAMP2'
            Overwrite scoring method to be used if desired. If `None`, the estimators scoring
            method will be used.
            Available scores are based on the variational approach for Markov processes [1]_ [2]_ :

            *  'VAMP1'  Sum of singular values of the symmetrized transition matrix [2]_ .
                        If the MSM is reversible, this is equal to the sum of transition
                        matrix eigenvalues, also called Rayleigh quotient [1]_ [3]_ .
            *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix [2]_ .
                        If the MSM is reversible, this is equal to the kinetic variance [4]_ .
            *  'VAMPE'  ...

        Returns
        -------

        """
        # TODO: implement for TICA too
        if test_model is None:
            test_model = self
        Uk = self.U[:, 0:self.dimension()]
        Vk = self.V[:, 0:self.dimension()]
        res = None
        if score_method == 'VAMP1' or score_method == 'VAMP2':
            A = spd_inv_sqrt(Uk.T.dot(test_model.C00).dot(Uk))
            B = Uk.T.dot(test_model.C0t).dot(Vk)
            C = spd_inv_sqrt(Vk.T.dot(test_model.Ctt).dot(Vk))
            ABC = mdot(A, B, C)
            if score_method == 'VAMP1':
                res = np.linalg.norm(ABC, ord='nuc')
            elif score_method == 'VAMP2':
                res = np.linalg.norm(ABC, ord='fro')**2
        elif score_method == 'VAMPE':
            Sk = np.diag(self.singular_values[0:self.dimension()])
            res = np.trace(2.0 * mdot(Vk, Sk, Uk.T, test_model.C0t) - mdot(Vk, Sk, Uk.T, test_model.C00, Uk, Sk, Vk.T, test_model.Ctt))
        else:
            raise ValueError('"score" should be one of VAMP1, VAMP2 or VAMPE')
        # add the contribution (+1) of the constant singular functions to the result
        assert res
        return res + 1


@fix_docs
class VAMP(StreamingEstimationTransformer, SerializableMixIn):
    r"""Variational approach for Markov processes (VAMP)"""

    __serialize_version = 0

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
        ncov_max : int, default=infinity
            limit the memory usage of the algorithm from [3]_ to an amount that corresponds
            to ncov_max additional copies of each correlation matrix

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        .. [3] Chan, T. F., Golub G. H., LeVeque R. J. 1979. Updating formulae and pairwiese algorithms for
           computing sample variances. Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
        """
        StreamingEstimationTransformer.__init__(self)

        # empty dummy model instance
        self._model = VAMPModel()
        self.set_params(lag=lag, dim=dim, scaling=scaling, right=right,
                        epsilon=epsilon, stride=stride, skip=skip, ncov_max=ncov_max)
        self._covar = None
        self._model.update_model_params(dim=dim, epsilon=epsilon)

    def _estimate(self, iterable, **kw):
        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=True, remove_data_mean=True, reversible=False,
                                       lag=self.lag, bessel=False, stride=self.stride, skip=self.skip, weights=None,
                                       ncov_max=self.ncov_max)
        indim = iterable.dimension()

        if isinstance(self.dim, int) and not self.dim <= indim:
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

        if self._covar is None:
            self._covar = LaggedCovariance(c00=True, c0t=True, ctt=True, remove_data_mean=True, reversible=False,
                                           lag=self.lag, bessel=False, stride=self.stride, skip=self.skip, weights=None,
                                           ncov_max=self.ncov_max)
        self._covar.partial_fit(iterable)
        self._model.update_model_params(mean_0=self._covar.mean,  # TODO: inefficient, fixme
                                        mean_t=self._covar.mean_tau,
                                        C00=self._covar.C00_,
                                        C0t=self._covar.C0t_,
                                        Ctt=self._covar.Ctt_)

        # self._used_data = self._covar._used_data
        self._estimated = False

        return self

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self._logger.debug("diagonalize covariance matrices")
        self.model._diagonalize(self.scaling)
        self._logger.debug("finished diagonalization.")
        self._estimated = True

    def dimension(self):
        return self._model.dimension()

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

    @property
    def singular_values(self):
        r"""Singular values of VAMP (usually denoted :math:`\sigma`)

        Returns
        -------
        singular values: 1-D np.array
        """
        return self._model.singular_values

    @property
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
    def cumvar(self):
        r"""Cumulative sum of the squared and normalized VAMP singular values

        Returns
        -------
        cumvar: 1D np.array
        """
        return self._model.cumvar

    @property
    def show_progress(self):
        if self._covar is None:
            return False
        else:
            return self._covar.show_progress

    @show_progress.setter
    def show_progress(self, value):
        if self._covar is not None:
            self._covar.show_progress = value

    def expectation(self, observables, statistics, lag_multiple=1, observables_mean_free=False,
                    statistics_mean_free=False):
        return self._model.expectation(statistics, observables, lag_multiple=lag_multiple,
                                       statistics_mean_free=statistics_mean_free,
                                       observables_mean_free=observables_mean_free)

    def cktest(self, n_observables=None, observables='psi', statistics='phi', mlags=10, n_jobs=1, show_progress=True,
               iterable=None):
        """

        Parameters
        ----------
        n_observables
        observables
        statistics
        mlags
        n_jobs
        show_progress
        iterable

        Returns
        -------

        """
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

        ck = VAMPChapmanKolmogorovValidator(self, self, observables, statistics, observables_mean_free,
                                            statistics_mean_free, mlags=mlags, n_jobs=n_jobs,
                                            show_progress=show_progress)

        if iterable is None:
            iterable = self.data_producer

        ck.estimate(iterable)
        return ck

    def score(self, test_data=None, score_method='VAMP2'):
        from pyemma._ext.sklearn.base import clone as clone_estimator
        est = clone_estimator(self)

        if test_data is None:
            return self.model.score(None, score_method=score_method)
        else:
            est.estimate(test_data)
            return self.model.score(est.model, score_method=score_method)


class VAMPChapmanKolmogorovValidator(LaggedModelValidator):
    __serialize_version = 0
    __serialize_fields = ('nsets', 'statistics', 'observables', 'observables_mean_free', 'statistics_mean_free')

    """
    
    Parameters
    ----------
    model : Model
        Model to be tested

    estimator : Estimator
        Parametrized Estimator that has produced the model

    statistics : np.ndarray((input_dimension, n_statistics)), optional
        Coefficients that express one or multiple statistics in
        the basis of the input features.
        This parameter can be None. In that case, this method
        returns the future expectation value of the observable(s).

    observables : np.ndarray((input_dimension, n_observables))
        Coefficients that express one or multiple observables in
        the basis of the input features.

    mlags : int or int-array, default=10
        multiples of lag times for testing the Model, e.g. range(10).
        A single int will trigger a range, i.e. mlags=10 maps to
        mlags=range(10). The setting None will choose mlags automatically
        according to the longest available trajectory
        Note that you need to be able to do a model prediction for each
        of these lag time multiples, e.g. the value 0 only make sense
        if _predict_observables(0) will work.

    conf : float, default = 0.95
        confidence interval for errors

    err_est : bool, default=False
        if the Estimator is capable of error calculation, will compute
        errors for each tau estimate. This option can be computationally
        expensive.

    n_jobs : int, default=1
        how many jobs to use during calculation

    show_progress : bool, default=True
        Show progressbars for calculation?
    """
    def __init__(self, model, estimator, observables, statistics, observables_mean_free, statistics_mean_free,
                 mlags=10, n_jobs=1, show_progress=True):
        LaggedModelValidator.__init__(self, model, estimator, mlags=mlags,
                                      n_jobs=n_jobs, show_progress=show_progress)
        self.statistics = statistics
        self.observables = observables
        self.observables_mean_free = observables_mean_free
        self.statistics_mean_free = statistics_mean_free
        if self.statistics is not None:
            self.nsets = min(self.observables.shape[1], self.statistics.shape[1])

    def _compute_observables(self, model, estimator, mlag=1):
        # for lag time 0 we return a matrix of nan, until the correct solution is implemented
        if mlag == 0 or model is None:
            if self.statistics is None:
                return np.zeros(self.observables.shape[1]) + np.nan
            else:
                return np.zeros((self.observables.shape[1], self.statistics.shape[1])) + np.nan
        else:
            return model.expectation(self.statistics, self.observables, lag_multiple=mlag,
                                     statistics_mean_free=self.statistics_mean_free,
                                     observables_mean_free=self.observables_mean_free)

    def _compute_observables_conf(self, model, estimator, mlag=1):
        raise NotImplementedError('estimation of confidence intervals not yet implemented for VAMP')
