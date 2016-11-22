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
from pyemma.util.types import is_float_vector, ensure_float_vector
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma._ext.variational.running_moments import running_covar

__all__ = ['CovarEstimator', ]

__author__ = 'paul, nueske'

class _KoopmanWeights(object):
    def __init__(self, u):
        self._u = u

    def weights(self, X):
        return X.dot(self._u[:-1]) + self._u[-1]


class CovarEstimator(StreamingEstimator):
    def __init__(self, xx=True, xy=False, yy=False, mean=None, remove_data_mean=False, symmetrize=False,
                 sparse_mode='auto', modify_data=False, lag=1, nsave=5, weight=None, stride=1,
                 skip=0, chunksize=None):

        super(CovarEstimator, self).__init__(chunksize=chunksize)

        self._rc = running_covar(xx=xx, xy=xy, yy=yy, mean=mean, remove_mean=remove_data_mean,
                                 symmetrize=symmetrize, time_lagged=False, sparse_mode=sparse_mode,
                                 modify_data=modify_data, nsave=nsave)

        if is_float_vector(weight):
            weight = ensure_float_vector(weight)
        self.set_params(xx=xx, xy=xy, yy=yy, mean=mean, remove_data_mean=remove_data_mean, symmetrize=symmetrize,
                        sparse_mode=sparse_mode, modify_data=modify_data, lag=lag, nsave=nsave,
                        weight=weight, stride=stride, skip=skip)

    def _compute_weight_series(self, X, it):
        if self.weight is None:
            return None
        elif if isinstance(weights, numbers.Real): # TODO: list of list + list, number
            pass
        elif isinstance(self.weight, np.ndarray):
            return self.weigth[it.itraj][it.pos:it:]
        else:
            return self.weight.weight(X)

    def _estimate(self, iterable, **kwargs):
        partial_fit = 'partial' in kw
        it = iterable.iterator(lag=self.lag, stride=self.stride, skip=self.skip, chunk = self.chunksize if not partial_fit else 0)


        self._init_covar(partial_fit, it.n_chunks)
        for X, Y in it:
            try:
                self._covar.add(X, Y)
            # counting chunks and log of eta
            self._progress_update(1, stage=0)

        with it:
            self._progress_register(it.n_chunks, "calculate mean+cov", 0)
            for data in it:
                if self.lag!=0:
                    X, Y = data
                else:
                    X, Y = data, None
                weight_series = self._compute_weight_series(X, it)
                try:
                    self._rc.add(X, Y, fixed_mean=self.mean, weights=weight_series)
                except MemoryError:
                    raise MemoryError('Covariance matrix does not fit into memory. '
                                      'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))

        self._model.update_model_params(mean=self._rc.mean_X(), # ?
                                        cov=self._rc.cov_XX(),
                                        cov_tau=self._rc.cov_XY())

@property
    def x_mean_0(self):
        return self._rc.mean_X()

    @property
    def x_mean_tau(self):
        return self._rc.mean_Y()

    @property
    def C_0(self):
        return self._rc.cov_XX()

    @property
    def C_tau(self):
        return self._rc.cov_XY()

    @property
    def nsave(self):
        if self.xx:
            return self._rc.storage_XX.nsave
        elif self.xy:
            return self._rc.storage_XY.nsave

    @nsave.setter
    def nsave(self, ns):
        if self.xx:
            if self._rc.storage_XX.nsave <= ns:
                self._rc.storage_XX.nsave = ns
        if self.xy:
            if self._rc.storage_XY.nsave <= ns:
                self._rc.storage_XY.nsave = ns