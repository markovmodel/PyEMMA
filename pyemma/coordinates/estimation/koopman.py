# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import scipy.linalg as scl
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma.coordinates.estimation.covariance import CovarEstimator
from pyemma._ext.variational.solvers.direct import sort_by_norm


__author__ = 'paul, nueske'


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


class _KoopmanWeights(object):
    def __init__(self, u, u_const):
        self._u = u
        self._u_const = u_const

    def weights(self, X):
        return X.dot(self._u) + self._u_const


class _KoopmanEstimator(StreamingEstimator):
    '''only for computing u
       The user-accessible way for computing K is TICA()
    '''

    def __init__(self, lag, epsilon=1e-6, stride=1, skip=0, chunksize=None):

        super(_KoopmanEstimator, self).__init__(chunksize=chunksize)

        self._covar = CovarEstimator(xx=True, xy=True, remove_data_mean=True, reversible=False,
                                     lag=lag, stride=stride, skip=skip)

        self.set_params(lag=lag, epsilon=epsilon, stride=stride, skip=skip)

    def partial_fit(self, X):
        from pyemma.coordinates import source
        self._covar.partial_fit(source(X))
        self._estimation_finished = False
        return self

    def _finish_estimation(self):
        s, Q = scl.eigh(self._covar.cov)
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
        R = np.dot(Q, np.diag(s ** -0.5))
        # Set the new correlation matrix:
        M = R.shape[1]
        K = np.dot(R.T, np.dot((self._covar.cov_tau), R))
        K = np.vstack((K, np.dot((self._covar.mean_tau - self._covar.mean), R)))
        ex1 = np.zeros((M + 1, 1))
        ex1[M, 0] = 1.0
        self._K = np.hstack((K, ex1))
        self._R = R

        self._estimation_finished = True

    def _estimate(self, iterable, **kwargs):
        self._covar.estimate(iterable, **kwargs)
        self._finish_estimation()
        return self

    @property
    def K(self):
        'Koopman operator on the modified basis (PC|1)'
        if not self._estimation_finished:
            self._finish_estimation()
        return self._K

    @property
    def u(self):
        'weights in the modified basis'
        return compute_u(self.K)

    @property
    def weights(self):
        'weights in the input basis (encapsulated in an object)'
        u_mod = self.u # in modified basis
        u_input = self._R.dot(u_mod[0:-1]) # in input basis
        u_input_const = u_mod[-1] - self.mean.dot(self._R.dot(u_mod[0:-1]))
        return _KoopmanWeights(u_input, u_input_const)

    @property
    def R(self):
        'weightening transformation'
        if not self._estimation_finished:
            self._finish_estimation()
        return self._R

    @property
    def mean(self):
        return self._covar.mean