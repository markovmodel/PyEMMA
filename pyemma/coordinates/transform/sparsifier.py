
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
Created on 22.07.2015

@author: marscher
'''

import numpy as np
from pyemma.coordinates.transform.transformer import Transformer


class Sparsifier(Transformer):
    r""" The Sparsifier checks its data source for constant features and removes them during transformation.

    Parameters
    ----------
    rtol : float
        relative tolerance to compare for constant features

    calc_mean : boolean
        should the mean of the data be calculated?

    lag : int
        needs to be set to the same value as in TICA, if force_eigen_values_le_one is True.

    force_eigenvalues_le_one : boolean
        Compute covariance matrix and time-lagged covariance matrix such
        that the generalized eigenvalues are always guaranteed to be <= 1.

    Notes
    -----
    The usage of the Sparsifier is recommended for contact features in MD data.
    Contacts which are never formed or never brake are being eliminated. This
    speeds-up further calculations.

    """

    def __init__(self, rtol=1e-2, calc_mean=True, lag=None,
                 force_eigenvalues_le_one=False):

        Transformer.__init__(self)
        self._varying_indices = None
        self._first_frame = None
        self._rtol = rtol
        self.calc_mean = calc_mean

        self._lag = lag
        self._force_eigenvalues_le_one = force_eigenvalues_le_one

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, value):
        self._rtol = value

    @property
    def varying_indices(self):
        return self._varying_indices

    def _param_init(self):
        self._varying_indices = []

        if self.calc_mean:
            self._N_mean = 0
            self.mu = np.zeros(self.data_producer.dimension(), dtype=np.float64)
            from pyemma.coordinates.transform.tica import TICA
            self._calc_mean = TICA._calc_mean

    def describe(self):
        return self.__class__.__name__ + 'dim: %s' % str(self.dimension()) if \
            self._parametrized else 'super-unknown'

    def dimension(self):
        if not self._parametrized:
            raise RuntimeError(
                "Sparsifier does not know its output dimension yet.")
        dim = len(self._varying_indices)
        return dim

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):

        if ipass == 0:
            if t == 0:
                self._first_frame = X[0]

            close = np.isclose(X, self._first_frame, rtol=self.rtol)
            not_close = np.logical_not(close)
            close_cols = np.argwhere(not_close)[:, 1]
            var_inds = np.unique(close_cols)
            self._varying_indices = np.union1d(var_inds, self._varying_indices)

            if self.calc_mean:
                self._N_mean = self._calc_mean(X, self.mu, self._N_mean,
                                               self.trajectory_length(itraj, stride),
                                               self._lag, stride, t,
                                               self._force_eigenvalues_le_one)

            if last_chunk:
                return True

        return False

    def _param_finish(self):
        self._varying_indices = np.array(self._varying_indices, dtype=int)
        if self.calc_mean:
            self.mu = self.mu[self._varying_indices]
        self._parametrized = True
        self._logger.warning("Detected and eliminated %i constant features"
                             % (self.data_producer.dimension() - self.dimension()))

    def _transform_array(self, X):
        return X[:, self._varying_indices]
