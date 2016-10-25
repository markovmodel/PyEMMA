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

from math import log

import numpy as np
import scipy.linalg as scl
from decorator import decorator

from pyemma._base.model import Model
from variational.estimators.running_moments import running_covar
from variational.solvers.direct import sort_by_norm
from pyemma.util.annotators import fix_docs, deprecated
from pyemma.util.linalg import eig_corr
from pyemma.util.reflection import get_default_args
from .transformer import StreamingTransformer


__all__ = ['TICA']

def compute_u(K):
    """
    Estimate an approximation of the ratio of stationary over empirical distribution from the basis.
    Parameters:
    -----------
    K0, ndarray(M+1, M+1),
        time-lagged correlation matrix for the whitened and padded data set.
    Returns:
    --------
    u : ndarray(M,)
        coefficients of the ratio stationary / empirical dist. from the whitened and expanded basis.
    """
    M = K.shape[0] - 1
    # Compute right and left eigenvectors:
    l, U = scl.eig(K.T)
    l, U = sort_by_norm(l, U)
    # Extract the eigenvector for eigenvalue one and normalize:
    u = np.real(U[:, 0])
    v = np.zeros(M+1)
    v[M] = 1.0
    u = u / np.dot(u, v)
    return u



class KoopmanModel(Model):
    def set_model_params(self, mean, cov, cov_tau, K, R):
        self.mean = mean
        self.cov = cov
        self.cov_tau = cov_tau
        self.K = K
        self.R = R

    def _diagonalize(self, method="a"):
        if method == "a":
            eigenvalues, eigenvectors = scl.eig(self.K)
        elif method == "s":
            eigenvalues, eigenvectors = scl.eigh(self.K)
        eigenvalues, eigenvectors = sort_by_norm(eigenvalues, eigenvectors)

        self.update_model_params(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def _decorrelate_basis(self, X):
        X_decor = np.dot((X - self.mean[None, :]), self.R)
        X_decor = np.hstack((X_decor, np.ones((X_decor.shape[0], 1))))
        return X_decor



@fix_docs
class Koopman(StreamingTransformer):
    r""" Time-lagged independent component analysis (TICA)"""

    def __init__(self, lag, dim=-1, epsilon=1e-6, stride=1, skip=0):
        r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
        lag : int
            lag time
        dim : int, optional, default -1
            Maximum number of significant independent components to use to reduce dimension of input data. -1 means
            all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.
        epsilon : float
            eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
        skip : int, default=0
            skip the first initial n frames per trajectory.


        """
        super(Koopman, self).__init__()

        # empty dummy model instance
        self._model = KoopmanModel()
        self.set_params(lag=lag, dim=dim, epsilon=epsilon, stride=stride, skip=skip)

    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        self._lag = new_tau

    def describe(self):
        try:
            dim = self.dimension()
        except AttributeError:
            dim = self.dim
        return "[Koopman, lag = %i; max. output dim. = %i]" % (self._lag, dim)

    def dimension(self):
        """ output dimension """
        if self.dim > -1:
            return self.dim
        elif self._estimated:
            return self.eigenvalues.shape[0]
        else:
            raise ValueError("Dimension depends on estimation results. Call estimate() first.")

    @property
    def mean(self):
        """ mean of input features """
        return self._model.mean

    @property
    def cov(self):
        """ covariance matrix of input data. """
        return self._model.cov

    @property
    def cov_tau(self):
        """ covariance matrix of time-lagged input data. """
        return self._model.cov_tau

    def _init_covar(self, n_chunks, xy=True):
        nsave = int(max(log(n_chunks, 2), 2))
        # in case we do a one shot estimation, we want to re-initialize running_covar
        self._logger.debug("using %s moments for %i chunks" % (nsave, n_chunks))
        self._covar = running_covar(xx=True, xy=xy, yy=False,
                                    remove_mean=False,
                                    symmetrize=False, nsave=nsave)

    def _init_estimate(self, iterable):
        indim = iterable.dimension()
        if not indim:
            raise ValueError("zero dimension from data source!")

        if not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))

        if self._logger_is_active(self._loglevel_DEBUG):
            self._logger.debug("Running Koopman with tau=%i; Estimating two covariance matrices"
                               " with dimension (%i, %i)" % (self._lag, indim, indim))

        if not any(iterable.trajectory_lengths(stride=self.stride, skip=self.lag+self.skip) > 0):
            raise ValueError("None single dataset [longest=%i] is longer than"
                                 " lag time [%i]." % (max(iterable.trajectory_lengths(self.stride, skip=self.skip)),
                                                      self.lag))

        self.logger.debug("will use {} total frames for {}".
                          format(iterable.trajectory_lengths(self.stride, skip=self.skip), self.name))

    def estimate(self, X, **kwargs):
        r"""
        Chunk-based parameterization of TICA. Iterates over all data and estimates
        the mean, covariance and time lagged covariance. Finally, the
        generalized eigenvalue problem is solved to determine
        the independent components.
        """
        return super(Koopman, self).estimate(X, **kwargs)

    def _estimate(self, iterable, **kw):
        indim = iterable.dimension()
        self._init_estimate(iterable)

        it = iterable.iterator(lag=self.lag, return_trajindex=False, chunk=self.chunksize, skip=self.skip)
        mean_x = np.zeros(indim)
        mean_y = np.zeros(indim)
        frames = 0
        with it:
            self._progress_register(it._n_chunks, "calculate means", 0)
            for X, Y in it:
                mean_x += np.sum(X, axis=0)
                mean_y += np.sum(Y, axis=0)
                frames += X.shape[0]
                # counting chunks and log of eta
                self._progress_update(1, stage=0)
        mean_x /= (1.0 * frames)
        mean_y /= (1.0 * frames)

        it = iterable.iterator(lag=self.lag, return_trajindex=False, chunk=self.chunksize, skip=self.skip)
        with it:
            self._progress_register(it._n_chunks, "calculate covariances", 1)
            self._init_covar(it._n_chunks)
            for X, Y in it:
                self._covar.add(X - mean_x[None, :], Y - mean_x[None, :])
                # counting chunks and log of eta
                self._progress_update(1, stage=1)

        s, Q = scl.eigh(self._covar.cov_XX())
        # Determine negative magnitudes:
        evmin = np.min(s)
        if evmin < 0:
            ep0 = np.maximum(self.epsilon, -evmin)
        else:
            ep0 = self.epsilon
        # Cut-off small or negative eigenvalues:
        s, Q = sort_by_norm(s, Q)
        ind = np.where(np.abs(s) > ep0)[0]
        s = s[ind]
        Q = Q[:, ind]
        for j in range(Q.shape[1]):
            jj = np.argmax(np.abs(Q[:, j]))
            Q[:, j] *= np.sign(Q[jj, j])
        # Compute the whitening transformation:
        R = np.dot(Q, np.diag(s**-0.5))
        # Set the new correlation matrix:
        M = R.shape[1]
        K = np.dot(R.T, np.dot((self._covar.cov_XY()), R))
        K = np.vstack((K, np.dot((mean_y - mean_x), R)))
        ex1 = np.zeros((M+1, 1))
        ex1[M, 0] = 1.0
        K = np.hstack((K, ex1))

        self._model.update_model_params(mean=mean_x,
                                        cov=self._covar.cov_XX(),
                                        cov_tau=self._covar.cov_XY(),
                                        K=K,
                                        R=R)
        # diagonalize with low rank approximation
        self._logger.debug("diagonalize K.")
        self._model._diagonalize()
        self._logger.debug("finished diagonalisation.")

        self._estimated = True

        return self._model

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
        X_decor = self._model._decorrelate_basis(X)
        Y = np.dot(X_decor, self.eigenvectors[:, 0:self.dimension()])
        return Y.astype(self.output_type())

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
        return -self.lag / np.log(np.abs(self.eigenvalues[1:]))

    @property
    def eigenvalues(self):
        r"""Eigenvalues of the TICA problem (usually denoted :math:`\lambda`

        Returns
        -------
        eigenvalues: 1D np.array
        """
        return self._model.eigenvalues

    @property
    def eigenvectors(self):
        r"""Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """
        return self._model.eigenvectors

    @property
    def model(self):
        return self._model

    @property
    def koopman_matrix(self):
        return self._model.K

class EquilibriumKoopman(Koopman):
    r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
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
            skip the first initial n frames per trajectory.

        """
    def __init__(self, lag, koopman_estimator, dim=-1, var_cutoff=0.95, kinetic_map=True, epsilon=1e-6, stride=1,
                 skip=0):
        default_var_cutoff = get_default_args(self.__init__)['var_cutoff']
        if dim != -1 and var_cutoff != default_var_cutoff:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either or.')

        super(Koopman, self).__init__()

        if dim > -1:
            var_cutoff = 1.0

        # empty dummy model instance
        self._model = KoopmanModel()
        self.set_params(lag=lag, dim=dim, koopman_estimator=koopman_estimator, var_cutoff=var_cutoff,
                        kinetic_map=kinetic_map, epsilon=epsilon, stride=stride, skip=skip)

    def _estimate(self, iterable, **kw):
        self._init_estimate(iterable)
        K0 = self.koopman_estimator._model.K
        u = compute_u(K0)
        it = iterable.iterator(lag=self.lag, return_trajindex=False, chunk=self.chunksize, skip=self.skip)
        with it:
            self._progress_register(it._n_chunks, "Re-weighting", 0)
            self._init_covar(it._n_chunks, xy=False)
            for X, Y in it:
                X = self.koopman_estimator._model._decorrelate_basis(X)
                w = np.dot(X, u)
                self._covar.add(X, weights=w)
                # counting chunks and log of eta
                self._progress_update(1, stage=0)
        C0_eq = self._covar.cov_XX()

        s, Q = scl.eigh(C0_eq)
        # Determine negative magnitudes:
        evmin = np.min(s)
        if evmin < 0:
            ep0 = np.maximum(self.epsilon, -evmin)
        else:
            ep0 = self.epsilon
        # Cut-off small or negative eigenvalues:
        s, Q = sort_by_norm(s, Q)
        ind = np.where(np.abs(s) > ep0)[0]
        s = s[ind]
        Q = Q[:, ind]
        for j in range(Q.shape[1]):
            jj = np.argmax(np.abs(Q[:, j]))
            Q[:, j] *= np.sign(Q[jj, j])
        # Determine whitening transformation at equilibrium:
        R = np.dot(Q, np.diag(s**-0.5))
        # Compute reversible Ctau and K:
        Ctau_eq = 0.5 * (np.dot(C0_eq, K0) + np.dot(K0.T, C0_eq))
        K = np.dot(R.T, np.dot(Ctau_eq, R))

        self._model.update_model_params(mean_pc=C0_eq[:, -1], cov_pc=C0_eq, cov_tau_pc=Ctau_eq,
                                        K=K, R=R, u=u)

        self._model._diagonalize(method="s")

        self._estimated = True

        return self._model

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
        X_decor = self.koopman_estimator._model._decorrelate_basis(X)
        Y = np.dot(X_decor, np.dot(self._model.R, self.eigenvectors[:, 0:self.dimension()]))
        return Y.astype(self.output_type())

    def reweighting_factors(self, skip=0, stride=1):
        it = self.data_producer.iterator(lag=0, return_trajindex=True, stride=stride, chunk=self.chunksize, skip=skip)
        weights = [np.zeros(trajlen) for trajlen in self.data_producer.trajectory_lengths(stride=stride, skip=skip)]
        with it:
            for ii, X in it:
                X = self.koopman_estimator._model._decorrelate_basis(X)
                w = np.dot(X, self._model.u)
                weights[ii][it.pos:(it.pos+w.shape[0])] = w
        return weights

    @property
    def non_equilibrium_estimator(self):
        return self.koopman_estimator

    @property
    def mean_pc(self):
        return self._model.mean_pc

    @property
    def cov_pc(self):
        return self._model.cov_pc

    @property
    def cov_tau_pc(self):
        return self._model.cov_tau_pc

    @property
    def mean(self):
        """ mean of input features """
        raise NotImplementedError("Equilibrium mean for original basis is not implemented.")

    @property
    def cov(self):
        """ covariance matrix of input data. """
        raise NotImplementedError("Equilibrium correlations for original basis are not implemented.")

    @property
    def cov_tau(self):
        """ covariance matrix of time-lagged input data. """
        raise NotImplementedError("Equilibrium correlations for original basis are not implemented.")