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

__all__ = ['VAMP', 'VAMPModel', 'VAMPChapmanKolmogorovValidator']


class VAMPModel(Model, SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = ('_U', '_singular_values', '_V', '_rank0', '_rankt', '_svd_performed')

    def set_model_params(self, mean_0, mean_t, C00, Ctt, C0t, dim, epsilon, scaling=None):
        self.mean_0 = mean_0
        self.mean_t = mean_t
        self.C00 = C00
        self.Ctt = Ctt
        self.C0t = C0t
        self._svd_performed = False
        self.dim = dim
        self.epsilon = epsilon
        self.scaling = scaling

    @property
    def scaling(self):
        """Scaling of projection. Can be None or 'kinetic map', 'km' """
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        if value not in (None, 'km', 'kinetic map'):
            raise ValueError('unexpected value (%s) of "scaling". Must be one of ("km", "kinetic map", None).' % value)
        self._scaling = value

    @property
    def U(self):
        "Tranformation matrix that represents the linear map from mean-free feature space to the space of left singular functions."
        if not self._svd_performed:
            self._diagonalize()
        return self._U

    @property
    def V(self):
        "Tranformation matrix that represents the linear map from mean-free feature space to the space of right singular functions."
        if not self._svd_performed:
            self._diagonalize()
        return self._V

    @property
    def singular_values(self):
        "The singular values of the half-weighted Koopman matrix"
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

    @staticmethod
    def _cumvar(singular_values):
        cumvar = np.cumsum(singular_values ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self):
        """ cumulative kinetic variance """
        return VAMPModel._cumvar(self.singular_values)

    @staticmethod
    def _dimension(rank0, rankt, dim, singular_values):
        """ output dimension """
        if dim is None or (isinstance(dim, float) and dim == 1.0):
            return min(rank0, rankt)
        if isinstance(dim, float):
            return np.count_nonzero(VAMPModel._cumvar(singular_values) >= dim)
        else:
            return np.min([rank0, rankt, dim])

    def dimension(self):
        """ output dimension """
        if self.C00 is None:  # no data yet
            if isinstance(self.dim, int):  # return user choice
                warnings.warn('Returning user-input for dimension, since this model has not yet been estimated.')
                return self.dim
            raise RuntimeError('Please call set_model_params prior using this method.')

        if not self._svd_performed:
            self._diagonalize()
        return self._dimension(self._rank0, self._rankt, self.dim, self.singular_values)

    def expectation(self, observables, statistics, lag_multiple=1, observables_mean_free=False, statistics_mean_free=False):
        r"""Compute future expectation of observable or covariance using the approximated Koopman operator.

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
            If true, coefficients in `observables` refer to the input
            features with feature means removed.
            If false, coefficients in `observables` refer to the
            unmodified input features.

        statistics_mean_free : bool, default=False
            If true, coefficients in `statistics` refer to the input
            features with feature means removed.
            If false, coefficients in `statistics` refer to the
            unmodified input features.

        Notes
        -----
        A "future expectation" of a observable g is the average of g computed
        over a time window that has the same total length as the input data
        from which the Koopman operator was estimated but is shifted
        by lag_multiple*tau time steps into the future (where tau is the lag
        time).

        It is computed with the equation:

        .. math::

            \mathbb{E}[g]_{\rho_{n}}=\mathbf{q}^{T}\mathbf{P}^{n-1}\mathbf{e}_{1}

        where

        .. math::

            P_{ij}=\sigma_{i}\langle\psi_{i},\phi_{j}\rangle_{\rho_{1}}

        and

        .. math::

            q_{i}=\langle g,\phi_{i}\rangle_{\rho_{1}}

        and :math:`\mathbf{e}_{1}` is the first canonical unit vector.


        A model prediction of time-lagged covariances between the
        observable f and the statistic g at a lag-time of lag_multiple*tau
        is computed with the equation:

        .. math::

            \mathrm{cov}[g,\,f;n\tau]=\mathbf{q}^{T}\mathbf{P}^{n-1}\boldsymbol{\Sigma}\mathbf{r}

        where :math:`r_{i}=\langle\psi_{i},f\rangle_{\rho_{0}}` and
        :math:`\boldsymbol{\Sigma}=\mathrm{diag(\boldsymbol{\sigma})}` .
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

    def _diagonalize(self):
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model.

        Parameters
        ----------
        scaling : None or string, default=None
            Scaling to be applied to the VAMP modes upon transformation
            * None: no scaling will be applied, variance of the singular
              functions is 1
            * 'kinetic map' or 'km': singular functions are scaled by
              singular value. Note that only the left singular functions
              induce a kinetic map.
        """

        L0, self._rank0 = spd_inv_sqrt(self.C00, epsilon=self.epsilon, return_rank=True)
        Lt, self._rankt = spd_inv_sqrt(self.Ctt, epsilon=self.epsilon, return_rank=True)
        A = L0.T.dot(self.C0t).dot(Lt)

        Uprime, s, Vprimeh = np.linalg.svd(A, compute_uv=True)
        self._singular_values = s

        self._L0 = L0
        self._Lt = Lt

        # don't pass any values calling this method again!!!
        m = VAMPModel._dimension(self._rank0, self._rankt, self.dim, self._singular_values)

        U = L0.dot(Uprime[:, :m])  # U in the paper singular_vectors_left
        V = Lt.dot(Vprimeh[:m, :].T)  # V in the paper singular_vectors_right

        # scale vectors
        if self.scaling is not None:
            U *= s[np.newaxis, 0:m]

        self._U = U
        self._V = V
        self._svd_performed = True

    def score(self, test_model=None, score_method='VAMP2'):
        """Compute the VAMP score for this model or the cross-validation score between self and a second model.

        Parameters
        ----------
        test_model : VAMPModel, optional, default=None

            If `test_model` is not None, this method computes the cross-validation score
            between self and `test_model`. It is assumed that self was estimated from
            the "training" data and `test_model` was estimated from the "test" data. The
            score is computed for one realization of self and `test_model`. Estimation
            of the average cross-validation score and partitioning of data into test and
            training part is not performed by this method.

            If `test_model` is None, this method computes the VAMP score for the model
            contained in self.

        score_method : str, optional, default='VAMP2'
            Available scores are based on the variational approach for Markov processes [1]_:

            *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the sum of
                        Koopman matrix eigenvalues, also called Rayleigh quotient [1]_.
            *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the kinetic variance [2]_ .
            *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                        the true Koopman operator up to an additive constant [1]_ .

        Returns
        -------
        score : float
            If `test_model` is not None, returns the cross-validation VAMP score between
            self and `test_model`. Otherwise return the selected VAMP-score of self.

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
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
    __serialize_fields = []

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
                  `n_components == min(n_samples, n_features)`
              * if dim is an integer >= 1, this number specifies the number
                of dimensions to keep. By default this will use the kinetic
                variance.
              * if dim is a float with ``0 < dim < 1``, select the number
                of dimensions such that the amount of kinetic variance
                that needs to be explained is greater than the percentage
                specified by dim.
          scaling : None or string
              Scaling to be applied to the VAMP order parameters upon transformation

              * None: no scaling will be applied, variance of the order parameters is 1
              * 'kinetic map' or 'km': order parameters are scaled by singular value
                Only the left singular functions induce a kinetic map.
                Therefore scaling='km' is only effective if `right` is False.
          right : boolean
              Whether to compute the right singular functions.
              If `right==True`, `get_output()` will return the right singular
              functions. Otherwise, `get_output()` will return the left singular
              functions.
              Beware that only `frames[tau:, :]` of each trajectory returned
              by `get_output()` contain valid values of the right singular
              functions. Conversely, only `frames[0:-tau, :]` of each
              trajectory returned by `get_output()` contain valid values of
              the left singular functions. The remaining frames might
              possibly be interpreted as some extrapolation.
          epsilon : float
              singular value cutoff. Singular values of :math:`C0` with
              norms <= epsilon will be cut off. The remaining number of
              singular values define the size of the output.
          stride: int, optional, default = 1
              Use only every stride-th time step. By default, every time step is used.
          skip : int, default=0
              skip the first initial n frames per trajectory.
          ncov_max : int, default=infinity
              limit the memory usage of the algorithm from [3]_ to an amount that corresponds
              to ncov_max additional copies of each correlation matrix

          Notes
          -----
          VAMP is a method for dimensionality reduction of Markov processes.

          The Koopman operator :math:`\mathcal{K}` is an integral operator
          that describes conditional future expectation values. Let
          :math:`p(\mathbf{x},\,\mathbf{y})` be the conditional probability
          density of visiting an infinitesimal phase space volume around
          point :math:`\mathbf{y}` at time :math:`t+\tau` given that the phase
          space point :math:`\mathbf{x}` was visited at the earlier time
          :math:`t`. Then the action of the Koopman operator on a function
          :math:`f` can be written as follows:

          .. math::

              \mathcal{K}f=\int p(\mathbf{x},\,\mathbf{y})f(\mathbf{y})\,\mathrm{dy}=\mathbb{E}\left[f(\mathbf{x}_{t+\tau}\mid\mathbf{x}_{t}=\mathbf{x})\right]

          The Koopman operator is defined without any reference to an
          equilibrium distribution. Therefore it is well-defined in
          situations where the dynamics is irreversible or/and non-stationary
          such that no equilibrium distribution exists.

          If we approximate :math:`f` by a linear superposition of ansatz
          functions :math:`\boldsymbol{\chi}` of the conformational
          degrees of freedom (features), the operator :math:`\mathcal{K}`
          can be approximated by a (finite-dimensional) matrix :math:`\mathbf{K}`.

          The approximation is computed as follows: From the time-dependent
          input features :math:`\boldsymbol{\chi}(t)`, we compute the mean
          :math:`\boldsymbol{\mu}_{0}` (:math:`\boldsymbol{\mu}_{1}`) from
          all data excluding the last (first) :math:`\tau` steps of every
          trajectory as follows:

          .. math::

              \boldsymbol{\mu}_{0}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\boldsymbol{\chi}(t)

              \boldsymbol{\mu}_{1}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\boldsymbol{\chi}(t)

          Next, we compute the instantaneous covariance matrices
          :math:`\mathbf{C}_{00}` and :math:`\mathbf{C}_{11}` and the
          time-lagged covariance matrix :math:`\mathbf{C}_{01}` as follows:

          .. math::

              \mathbf{C}_{00}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

              \mathbf{C}_{11}	:=\frac{1}{T-\tau}\sum_{t=\tau}^{T}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]

              \mathbf{C}_{01}	:=\frac{1}{T-\tau}\sum_{t=0}^{T-\tau}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]\left[\boldsymbol{\chi}(t+\tau)-\boldsymbol{\mu}_{1}\right]

          The Koopman matrix is then computed as follows:

          .. math::

              \mathbf{K}=\mathbf{C}_{00}^{-1}\mathbf{C}_{01}

          It can be shown [1]_ that the leading singular functions of the
          half-weighted Koopman matrix

          .. math::

              \bar{\mathbf{K}}:=\mathbf{C}_{00}^{-\frac{1}{2}}\mathbf{C}_{01}\mathbf{C}_{11}^{-\frac{1}{2}}

          encode the best reduced dynamical model for the time series.

          The singular functions can be computed by first performing the
          singular value decomposition

          .. math::

              \bar{\mathbf{K}}=\mathbf{U}^{\prime}\mathbf{S}\mathbf{V}^{\prime}

          and then mapping the input conformation to the left singular
          functions :math:`\boldsymbol{\psi}` and right singular
          functions :math:`\boldsymbol{\phi}` as follows:

          .. math::

              \boldsymbol{\psi}(t):=\mathbf{U}^{\prime\top}\mathbf{C}_{00}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{0}\right]

              \boldsymbol{\phi}(t):=\mathbf{V}^{\prime\top}\mathbf{C}_{11}^{-\frac{1}{2}}\left[\boldsymbol{\chi}(t)-\boldsymbol{\mu}_{1}\right]


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
        self._model.update_model_params(dim=dim, epsilon=epsilon, scaling=scaling)

    def _estimate(self, iterable, **kw):
        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=True, remove_data_mean=True, reversible=False,
                                       lag=self.lag, bessel=False, stride=self.stride, skip=self.skip, weights=None,
                                       ncov_max=self.ncov_max)
        indim = iterable.dimension()

        if isinstance(self.dim, int) and not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))

        if self._logger_is_active(self._loglevel_DEBUG):
            self.logger.debug("Running VAMP with tau=%i; Estimating two covariance matrices"
                               " with dimension (%i, %i)" % (self._lag, indim, indim))

        self._covar.estimate(iterable, **kw)
        self._model.update_model_params(mean_0=self._covar.mean,
                                        mean_t=self._covar.mean_tau,
                                        C00=self._covar.C00_,
                                        C0t=self._covar.C0t_,
                                        Ctt=self._covar.Ctt_)
        self.model._diagonalize()
        # if the previous estimation was a partial_fit, we might have a running covar object, which we can safely omit now.
        if '_covar' in self.__serialize_fields:
            self.__serialize_fields.remove('_covar')

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

        self._estimated = False
        if '_covar' not in self.__serialize_fields:
            self.__serialize_fields.append('_covar')
        return self

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
            If `self.right` is True, projection will be on the right singular
            functions. Otherwise, projection will be on the left singular
            functions.
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
        r"""Singular values of the half-weighted Koopman matrix (usually denoted :math:`\sigma`)

        Returns
        -------
        singular values: 1-D np.array
        """
        return self._model.singular_values

    @property
    def singular_vectors_right(self):
        r"""Tranformation matrix that represents the linear map from feature space to the space of right singular functions.

        Notes
        -----
        Right "singular vectors" V of the VAMP problem (equation 13 in [1]_), columnwise

        Returns
        -------
        vectors: 2-D ndarray
            Coefficients that express the right singular functions in the
            basis of mean-free input features.

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        """
        return self._model.V

    @property
    def singular_vectors_left(self):
        r"""Tranformation matrix that represents the linear map from feature space to the space of left singular functions.

        Notes
        -----
        Left "singular vectors" U of the VAMP problem (equation 13 in [1]_), columnwise

        Returns
        -------
        vectors: 2-D ndarray
            Coefficients that express the left singular functions in the
            basis of mean-free input features.

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        """
        return self._model.U

    @property
    def cumvar(self):
        r"""Cumulative sum of the squared and normalized singular values

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
        r"""Compute future expectation of observable or covariance using the approximated Koopman operator.

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
            If true, coefficients in `observables` refer to the input
            features with feature means removed.
            If false, coefficients in `observables` refer to the
            unmodified input features.

        statistics_mean_free : bool, default=False
            If true, coefficients in `statistics` refer to the input
            features with feature means removed.
            If false, coefficients in `statistics` refer to the
            unmodified input features.

        Notes
        -----
        A "future expectation" of a observable g is the average of g computed
        over a time window that has the same total length as the input data
        from which the Koopman operator was estimated but is shifted
        by lag_multiple*tau time steps into the future (where tau is the lag
        time).

        It is computed with the equation:

        .. math::

            \mathbb{E}[g]_{\rho_{n}}=\mathbf{q}^{T}\mathbf{P}^{n-1}\mathbf{e}_{1}

        where

        .. math::

            P_{ij}=\sigma_{i}\langle\psi_{i},\phi_{j}\rangle_{\rho_{1}}

        and

        .. math::

            q_{i}=\langle g,\phi_{i}\rangle_{\rho_{1}}

        and :math:`\mathbf{e}_{1}` is the first canonical unit vector.


        A model prediction of time-lagged covariances between the
        observable f and the statistic g at a lag-time of lag_multiple*tau
        is computed with the equation:

        .. math::

            \mathrm{cov}[g,\,f;n\tau]=\mathbf{q}^{T}\mathbf{P}^{n-1}\boldsymbol{\Sigma}\mathbf{r}

        where :math:`r_{i}=\langle\psi_{i},f\rangle_{\rho_{0}}` and
        :math:`\boldsymbol{\Sigma}=\mathrm{diag(\boldsymbol{\sigma})}` .
        """
        return self._model.expectation(observables, statistics, lag_multiple=lag_multiple,
                                       statistics_mean_free=statistics_mean_free,
                                       observables_mean_free=observables_mean_free)

    def cktest(self, n_observables=None, observables='phi', statistics='psi', mlags=10, n_jobs=1, show_progress=True,
               iterable=None):
        r"""Do the Chapman-Kolmogorov test by computing predictions for higher lag times and by performing estimations at higher lag times.

        Notes
        -----

        This method computes two sets of time-lagged covariance matrices

        * estimates at higher lag times :

          .. math::

              \left\langle \mathbf{K}(n\tau)g_{i},f_{j}\right\rangle_{\rho_{0}}

          where :math:`\rho_{0}` is the empirical distribution implicitly defined
          by all data points from time steps 0 to T-tau in all trajectories,
          :math:`\mathbf{K}(n\tau)` is a rank-reduced Koopman matrix estimated
          at the lag-time n*tau and g and f are some functions of the data.
          Rank-reduction of the Koopman matrix is controlled by the `dim`
          parameter of :func:`vamp <pyemma.coordinates.vamp>`.

        * predictions at higher lag times :

          .. math::

              \left\langle \mathbf{K}^{n}(\tau)g_{i},f_{j}\right\rangle_{\rho_{0}}

          where :math:`\mathbf{K}^{n}` is the n'th power of the rank-reduced
          Koopman matrix contained in self.


        The Champan-Kolmogorov test is to compare the predictions to the
        estimates.

        Parameters
        ----------
        n_observables : int, optional, default=None
            Limit the number of default observables (and of default statistics)
            to this number.
            Only used if `observables` are None or `statistics` are None.

        observables : np.ndarray((input_dimension, n_observables)) or 'phi'
            Coefficients that express one or multiple observables :math:`g`
            in the basis of the input features.
            This parameter can be 'phi'. In that case, the dominant
            right singular functions of the Koopman operator estimated
            at the smallest lag time are used as default observables.

        statistics : np.ndarray((input_dimension, n_statistics)) or 'psi'
            Coefficients that express one or multiple statistics :math:`f`
            in the basis of the input features.
            This parameter can be 'psi'. In that case, the dominant
            left singular functions of the Koopman operator estimated
            at the smallest lag time are used as default statistics.

        mlags : int or int-array, default=10
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10).
            Note that you need to be able to do a model prediction for each
            of these lag time multiples, e.g. the value 0 only make sense
            if model.expectation(lag_multiple=0) will work.

        n_jobs : int, default=1
            how many jobs to use during calculation

        show_progress : bool, default=True
            Show progressbars for calculation?

        iterable : any data format that `pyemma.coordinates.vamp()` accepts as input, optional
            It `iterable` is None, the same data source with which VAMP
            was initialized will be used for all estimation.
            Otherwise, all estimates (not predictions) from data will be computed
            from the data contained in `iterable`.

        Returns
        -------
        vckv : :class:`VAMPChapmanKolmogorovValidator <pyemma.coordinates.transform.VAMPChapmanKolmogorovValidator>`
            Contains the estimated and the predicted covarince matrices.
            The object can be plotted with :func:`plot_cktest <pyemma.plots.plot_cktest>` with the option `y01=False`.
        """
        if n_observables is not None:
            if n_observables > self.dimension():
                warnings.warn('Selected singular functions as observables but dimension '
                              'is lower than requested number of observables.')
                n_observables = self.dimension()
        else:
            n_observables = self.dimension()

        if isinstance(observables, str) and observables == 'phi':
            observables = self.singular_vectors_right[:, 0:n_observables]
            observables_mean_free = True
        else:
            ensure_ndarray(observables, ndim=2)
            observables_mean_free = False

        if isinstance(statistics, str) and statistics == 'psi':
            statistics = self.singular_vectors_left[:, 0:n_observables]
            statistics_mean_free = True
        else:
            ensure_ndarray_or_None(statistics, ndim=2)
            statistics_mean_free = False

        ck = VAMPChapmanKolmogorovValidator(self.model, self, observables, statistics, observables_mean_free,
                                            statistics_mean_free, mlags=mlags, n_jobs=n_jobs,
                                            show_progress=show_progress)

        if iterable is None:
            iterable = self.data_producer

        ck.estimate(iterable)
        return ck

    def score(self, test_data=None, score_method='VAMP2'):
        """Compute the VAMP score for this model or the cross-validation score between self and a second model estimated form different data.

        Parameters
        ----------
        test_data : any data format that `pyemma.coordinates.vamp()` accepts as input

            If `test_data` is not None, this method computes the cross-validation score
            between self and a VAMP model estimated from `test_data`. It is assumed that
            self was estimated from the "training" data and `test_data` is the test data.
            The score is computed for one realization of self and `test_data`. Estimation
            of the average cross-validation score and partitioning of data into test and
            training part is not performed by this method.

            If `test_data` is None, this method computes the VAMP score for the model
            contained in self.

            The model that is estimated from `test_data` will inherit all hyperparameters
            from self.

        score_method : str, optional, default='VAMP2'
            Available scores are based on the variational approach for Markov processes [1]_:

            *  'VAMP1'  Sum of singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the sum of
                        Koopman matrix eigenvalues, also called Rayleigh quotient [1]_.
            *  'VAMP2'  Sum of squared singular values of the half-weighted Koopman matrix [1]_ .
                        If the model is reversible, this is equal to the kinetic variance [2]_ .
            *  'VAMPE'  Approximation error of the estimated Koopman operator with respect to
                        the true Koopman operator up to an additive constant [1]_ .

        Returns
        -------
        score : float
            If `test_data` is not None, returns the cross-validation VAMP score between
            self and the model estimated from `test_data`. Otherwise return the selected
            VAMP-score of self.

        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        """
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

    def __init__(self, model, estimator, observables, statistics, observables_mean_free, statistics_mean_free,
                 mlags=10, n_jobs=1, show_progress=True):
        r"""
         Note
         ----
         It is recommended that you create this object by calling the
         `cktest` method of a VAMP object created with
         :func:`vamp <pyemma.coordinates.vamp>`.

         Parameters
         ----------
         model : Model
             Model with the smallest lag time. Is used to make predictions
             for larger lag times.

         estimator : Estimator
             Parametrized Estimator that has produced the model.
             Is used as a prototype for estimating models at higher lag times.

         observables : np.ndarray((input_dimension, n_observables))
             Coefficients that express one or multiple observables in
             the basis of the input features.

         statistics : np.ndarray((input_dimension, n_statistics))
             Coefficients that express one or multiple statistics in
             the basis of the input features.

         observables_mean_free : bool, default=False
             If true, coefficients in `observables` refer to the input
             features with feature means removed.
             If false, coefficients in `observables` refer to the
             unmodified input features.

         statistics_mean_free : bool, default=False
             If true, coefficients in `statistics` refer to the input
             features with feature means removed.
             If false, coefficients in `statistics` refer to the
             unmodified input features.

         mlags : int or int-array, default=10
             multiples of lag times for testing the Model, e.g. range(10).
             A single int will trigger a range, i.e. mlags=10 maps to
             mlags=range(10).
             Note that you need to be able to do a model prediction for each
             of these lag time multiples, e.g. the value 0 only make sense
             if model.expectation(lag_multiple=0) will work.

         n_jobs : int, default=1
             how many jobs to use during calculation

         show_progress : bool, default=True
             Show progressbars for calculation?

         Notes
         -----
         The object can be plotted with :func:`plot_cktest <pyemma.plots.plot_cktest>`
         with the option `y01=False`.
         """
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
            return model.expectation(statistics=self.statistics, observables=self.observables, lag_multiple=mlag,
                                     statistics_mean_free=self.statistics_mean_free,
                                     observables_mean_free=self.observables_mean_free)

    def _compute_observables_conf(self, model, estimator, mlag=1):
        raise NotImplementedError('estimation of confidence intervals not yet implemented for VAMP')
