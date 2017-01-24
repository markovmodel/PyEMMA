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
from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.estimators.running_moments import running_covar


__all__ = ['LaggedCovariance']

__author__ = 'paul, nueske'


class LaggedCovariance(StreamingEstimator, ProgressReporter):
    r"""Compute lagged covariances between time series.

     Parameters
     ----------
     c00 : bool, optional, default=True
         compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
     c0t : bool, optional, default=False
         compute lagged correlations. Does not work with lag==0.
     ctt : bool, optional, default=False
         compute instantaneous correlations over the second part of the data. Does not work with lag==0.
     remove_constant_mean : ndarray(N,), optional, default=None
         substract a constant vector of mean values from time series.
     remove_data_mean : bool, optional, default=False
         substract the sample mean from the time series (mean-free correlations).
     reversible : bool, optional, default=False
         symmetrize correlations.
     bessel : bool, optional, default=True
         use Bessel's correction for correlations in order to use an unbiased estimator
     sparse_mode : str, optional, default='auto'
         one of:
             * 'dense' : always use dense mode
             * 'auto' : automatic
             * 'sparse' : always use sparse mode if possible
     modify_data : bool, optional, default=False
         If remove_data_mean=True, the mean will be removed in the input data, without creating an independent copy.
         This option is faster but should only be selected if the input data is not used elsewhere.
     lag : int, optional, default=0
         lag time. Does not work with c0t=True or ctt=True.
     weights : trajectory weights.
         one of:
             * None :    all frames have weight one.
             * float :   all frames have the same specified weight.
             * object:   an object that possesses a .weight(X) function in order to assign weights to every
                         time step in a trajectory X.
     stride: int, optional, default = 1
         Use only every stride-th time step. By default, every time step is used.
     skip : int, optional, default=0
         skip the first initial n frames per trajectory.
     chunksize : int, optional, default=None
         The chunk size at which the input files are being processed.

     """
    def __init__(self, c00=True, c0t=False, ctt=False, remove_constant_mean=None, remove_data_mean=False, reversible=False,
                 bessel=True, sparse_mode='auto', modify_data=False, lag=0, weights=None, stride=1, skip=0,
                 chunksize=None, ncov_max=float('inf')):
        super(LaggedCovariance, self).__init__(chunksize=chunksize)

        if (c0t or ctt) and lag == 0:
            raise ValueError("lag must be positive if c0t=True or ctt=True")
        if is_float_vector(weights):
            weights = ensure_float_vector(weights)
        if remove_constant_mean is not None and remove_data_mean:
            raise ValueError('Subtracting the data mean and a constant vector simultaneously is not supported.')
        if remove_constant_mean is not None:
            remove_constant_mean = ensure_float_vector(remove_constant_mean)
        self.set_params(c00=c00, c0t=c0t, ctt=ctt, remove_constant_mean=remove_constant_mean,
                        remove_data_mean=remove_data_mean, reversible=reversible,
                        sparse_mode=sparse_mode, modify_data=modify_data, lag=lag,
                        bessel=bessel,
                        weights=weights, stride=stride, skip=skip, ncov_max=ncov_max)

        self._rc = None
        self._used_data = 0

    def _compute_weight_series(self, X, it):
        if self.weights is None:
            return None
        elif isinstance(self.weights, numbers.Real):
            return self.weights
        else:
            return self.weights.weights(X)

    def _init_covar(self, partial_fit, n_chunks):
        nsave = min(int(max(log(n_chunks, 2), 2)), self.ncov_max)
        if self._rc is not None and partial_fit:
            # check storage size vs. n_chunks of the new iterator
            old_nsave = self.nsave
            if old_nsave < nsave:
                self.logger.info("adapting storage size")
                self.nsave = nsave
        else: # in case we do a one shot estimation, we want to re-initialize running_covar
            self._logger.debug("using %s moments for %i chunks" % (nsave, n_chunks))
            self._rc = running_covar(xx=self.c00, xy=self.c0t, yy=self.ctt,
                                     remove_mean=self.remove_data_mean, symmetrize=self.reversible,
                                     sparse_mode=self.sparse_mode, modify_data=self.modify_data, nsave=nsave)

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

        # TODO: we could possibly optimize the case lag>0 and c0t=False using skip.
        # Access how much iterator hassle this would be.
        with it:
            self._progress_register(it.n_chunks, "calculate covariances", 0)
            self._init_covar(partial_fit, it.n_chunks)
            for data in it:
                if self.lag!=0:
                    X, Y = data
                else:
                    X, Y = data, None

                weight_series = self._compute_weight_series(X, it)

                if self.remove_constant_mean is not None:
                    X = X - self.remove_constant_mean[np.newaxis, :]
                    if Y is not None:
                        Y = Y - self.remove_constant_mean[np.newaxis, :]

                try:
                    self._rc.add(X, Y, weights=weight_series)
                except MemoryError:
                    raise MemoryError('Covariance matrix does not fit into memory. '
                                      'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))
                self._progress_update(1, stage=0)

        if partial_fit:
            self._used_data += len(it)

    def partial_fit(self, X):
        """ incrementally update the estimates

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.
        """
        from pyemma.coordinates import source

        self._estimate(source(X), partial=True)
        self._estimated = True

        return self

    @property
    def mean(self):
        self._check_estimated()
        return self._rc.mean_X()

    @property
    def mean_tau(self):
        self._check_estimated()
        return self._rc.mean_Y()

    @property
    def cov(self):
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    def cov_tau(self):
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def nsave(self):
        if self.c00:
            return self._rc.storage_XX.nsave
        elif self.c0t:
            return self._rc.storage_XY.nsave

    @nsave.setter
    def nsave(self, ns):
        if self.c00:
            if self._rc.storage_XX.nsave <= ns:
                self._rc.storage_XX.nsave = ns
        if self.c0t:
            if self._rc.storage_XY.nsave <= ns:
                self._rc.storage_XY.nsave = ns
