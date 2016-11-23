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
import numbers
from math import log
from pyemma.util.types import is_float_vector, ensure_float_vector
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma.coordinates.data._base.iterable import Iterable
from pyemma._base.logging import Loggable
from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.running_moments import running_covar


__all__ = ['CovarEstimator', ]

__author__ = 'paul, nueske'

class _KoopmanWeights(object):
    def __init__(self, u):
        self._u = u

    def weights(self, X):
        return X.dot(self._u[:-1]) + self._u[-1]


class CovarEstimator(StreamingEstimator, ProgressReporter, Loggable):
    def __init__(self, xx=True, xy=False, yy=False, remove_constant_mean=None, remove_data_mean=False, symmetrize=False,
                 sparse_mode='auto', modify_data=False, lag=0, weight=None, stride=1, skip=0, chunksize=None):

        super(CovarEstimator, self).__init__(chunksize=chunksize)

        if is_float_vector(weight):
            weight = ensure_float_vector(weight)
        self.set_params(xx=xx, xy=xy, yy=yy, remove_constant_mean=remove_constant_mean,
                        remove_data_mean=remove_data_mean, symmetrize=symmetrize,
                        sparse_mode=sparse_mode, modify_data=modify_data, lag=lag,
                        weight=weight, stride=stride, skip=skip)

    def _compute_weight_series(self, X, it):
        if self.weight is None:
            return None
        elif isinstance(self.weight, numbers.Real): # TODO: list of list + list, number
            pass
        elif isinstance(self.weight, np.ndarray):
            return self.weigth[it.itraj][it.pos:it:]
        else:
            return self.weight.weight(X)

    def _init_covar(self, partial_fit, n_chunks):
        nsave = int(max(log(n_chunks, 2), 2))
        # in case we do a one shot estimation, we want to re-initialize running_covar
        if hasattr(self, '_rc') and partial_fit:
            # check storage size vs. n_chunks of the new iterator
            old_nsave = self.nsave
            if old_nsave < nsave:
                self.logger.info("adapting storage size")
                self.nsave = nsave
        else:
            self._logger.debug("using %s moments for %i chunks" % (nsave, n_chunks))
            self._rc = running_covar(xx=self.xx, xy=self.xy, yy=self.yy, #mean=self.remove_constant_mean,
                                     remove_mean=self.remove_data_mean, symmetrize=self.symmetrize, time_lagged=False,
                                     sparse_mode=self.sparse_mode, modify_data=self.modify_data, nsave=nsave)
    def partial_fit(self, X):
        """ incrementally update the estimates

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.
        """
        if isinstance(X, Iterable):
            iterable = X
        else:
            from pyemma.coordinates import source
            iterable = source(X)

        self._estimate(iterable, partial=True)

        return self

    def _estimate(self, iterable, **kw):
        partial_fit = 'partial' in kw
        indim = iterable.dimension()
        if not indim:
            raise ValueError("zero dimension from data source!")

        if not any(iterable.trajectory_lengths(stride=self.stride, skip=self.lag+self.skip) > 0):
            if partial_fit:
                self.logger.warn("Could not use data passed to partial_fit(), "
                                 "because no single data set [longest=%i] is longer than lag+skip [%i]"
                                 % (max(iterable.trajectory_lengths(self.stride, skip=self.skip)), self.lag+self.skip))
                return self
            else:
                raise ValueError("None single dataset [longest=%i] is longer than"
                                 " lag+skip [%i]." % (max(iterable.trajectory_lengths(self.stride, skip=self.skip)),
                                                      self.lag+self.skip))

        self.logger.debug("will use {} total frames for {}".
                          format(iterable.trajectory_lengths(self.stride, skip=self.skip), self.name))

        it = iterable.iterator(lag=self.lag, return_trajindex=False, stride=self.stride, skip=self.skip, chunk = self.chunksize if not partial_fit else 0)

        with it:
            self._progress_register(it.n_chunks, "calculate covariances", 0)
            self._init_covar(partial_fit, it.n_chunks)
            for data in it:
                if self.lag!=0:
                    X, Y = data
                else:
                    X, Y = data, None
                weight_series = self._compute_weight_series(X, it)
                try:
                    self._rc.add(X, Y, weights=weight_series) #fixed_mean=self.remove_constant_mean,
                except MemoryError:
                    raise MemoryError('Covariance matrix does not fit into memory. '
                                      'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))
                self._progress_update(1, stage=0)

        if partial_fit:
            if not hasattr(self, "_used_data"):
                self._used_data = 0
            self._used_data += len(it)

    @property
    def mean(self):
        return self._rc.mean_X()

    @property
    def mean_tau(self):
        return self._rc.mean_Y()

    @property
    def cov(self):
        return self._rc.cov_XX()

    @property
    def cov_tau(self):
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