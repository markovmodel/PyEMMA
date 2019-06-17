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


import numpy as np
import numbers
from math import log

from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.util.annotators import deprecated
from pyemma.util.types import is_float_vector, ensure_float_vector
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.estimators.running_moments import running_covar


__all__ = ['LaggedCovariance']

__author__ = 'paul, nueske'


class LaggedCovariance(StreamingEstimator, SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = []

    r"""Compute lagged covariances between time series.

     Parameters
     ----------
     c00 : bool, optional, default=True
         compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
         Makes the C00_ attribute available.
     c0t : bool, optional, default=False
         compute lagged correlations. Does not work with lag==0.
         Makes the C0t_ attribute available.
     ctt : bool, optional, default=False
         compute instantaneous correlations over the time-shifted chunks of the data. Does not work with lag==0.
         Makes the Ctt_ attribute available.
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
             * list of arrays: ....

     stride: int, optional, default = 1
         Use only every stride-th time step. By default, every time step is used.
     skip : int, optional, default=0
         skip the first initial n frames per trajectory.
     chunksize : deprecated, default=NotImplemented
         The chunk size should now be set during estimation.
     column_selection: ndarray(k, dtype=int) or None
         Indices of those columns that are to be computed. If None, all columns are computed.
     diag_only: bool
         If True, the computation is restricted to the diagonal entries (autocorrelations) only.

     """
    def __init__(self, c00=True, c0t=False, ctt=False, remove_constant_mean=None, remove_data_mean=False, reversible=False,
                 bessel=True, sparse_mode='auto', modify_data=False, lag=0, weights=None, stride=1, skip=0,
                 chunksize=NotImplemented, ncov_max=float('inf'), column_selection=None, diag_only=False):
        super(LaggedCovariance, self).__init__()
        if chunksize is not NotImplemented:
            import warnings
            from pyemma.util.exceptions import PyEMMA_DeprecationWarning
            warnings.warn('passed deprecated argument chunksize to LaggedCovariance. Will be ignored!',
                          category=PyEMMA_DeprecationWarning)

        if (c0t or ctt) and lag == 0:
            raise ValueError("lag must be positive if c0t=True or ctt=True")

        if remove_constant_mean is not None and remove_data_mean:
            raise ValueError('Subtracting the data mean and a constant vector simultaneously is not supported.')
        if remove_constant_mean is not None:
            remove_constant_mean = ensure_float_vector(remove_constant_mean)
        if column_selection is not None and diag_only:
            raise ValueError('Computing only parts of the diagonal is not supported.')
        if diag_only and sparse_mode is not 'dense':
            if sparse_mode is 'sparse':
                self.logger.warning('Computing diagonal entries only is not implemented for sparse mode. Switching to dense mode.')
            sparse_mode = 'dense'
        self.set_params(c00=c00, c0t=c0t, ctt=ctt, remove_constant_mean=remove_constant_mean,
                        remove_data_mean=remove_data_mean, reversible=reversible,
                        sparse_mode=sparse_mode, modify_data=modify_data, lag=lag,
                        bessel=bessel,
                        weights=weights, stride=stride, skip=skip, ncov_max=ncov_max,
                        column_selection=column_selection, diag_only=diag_only)

        self._rc = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        from pyemma.coordinates.data import DataInMemory
        import types

        if is_float_vector(value):
            value = DataInMemory(value)
        elif isinstance(value, (list, tuple)):
            value = DataInMemory(value)
        elif isinstance(value, numbers.Integral):
            value = float(value) if value is not None else 1.0
        elif hasattr(value, 'weights') and type(getattr(value, 'weights')) == types.MethodType:
            from pyemma.coordinates.data._base.transformer import StreamingTransformer
            class compute_weights_streamer(StreamingTransformer):
                def __init__(self, func):
                    super(compute_weights_streamer, self).__init__()
                    self.func = func
                def dimension(self):
                    return 1
                def _transform_array(self, X):
                    return self.func.weights(X)
                def describe(self): pass

            value = compute_weights_streamer(value)

        self._weights = value

    def _init_covar(self, partial_fit, n_chunks):
        nsave = min(int(max(log(n_chunks, 2), 2)), self.ncov_max)
        if self._rc is not None and partial_fit:
            # check storage size vs. n_chunks of the new iterator
            old_nsave = self.nsave
            if old_nsave < nsave:
                self.logger.info("adapting storage size")
                self.nsave = nsave
        else: # in case we do a one shot estimation, we want to re-initialize running_covar
            self.logger.debug("using %s moments for %i chunks", nsave, n_chunks)
            self._rc = running_covar(xx=self.c00, xy=self.c0t, yy=self.ctt,
                                     remove_mean=self.remove_data_mean, symmetrize=self.reversible,
                                     sparse_mode=self.sparse_mode, modify_data=self.modify_data,
                                     column_selection=self.column_selection, diag_only=self.diag_only,
                                     nsave=nsave)

    def _estimate(self, iterable, partial_fit=False):
        indim = iterable.dimension()
        if not indim:
            raise ValueError("zero dimension from data source!")

        if not any(iterable.trajectory_lengths(stride=self.stride, skip=self.lag+self.skip) > 0):
            if partial_fit:
                self.logger.warning("Could not use data passed to partial_fit(), "
                                 "because no single data set [longest=%i] is longer than lag+skip [%i]",
                                 max(iterable.trajectory_lengths(self.stride, skip=self.skip)), self.lag+self.skip)
                return self
            else:
                raise ValueError("None single dataset [longest=%i] is longer than"
                                 " lag+skip [%i]." % (max(iterable.trajectory_lengths(self.stride, skip=self.skip)),
                                                      self.lag+self.skip))

        self.logger.debug("will use %s total frames for %s",
                          iterable.trajectory_lengths(self.stride, skip=self.skip), self.name)

        chunksize = 0 if partial_fit else iterable.chunksize
        it = iterable.iterator(lag=self.lag, return_trajindex=False, stride=self.stride, skip=self.skip,
                               chunk=chunksize)
        # iterator over input weights
        if hasattr(self.weights, 'iterator'):
            if hasattr(self.weights, '_transform_array'):
                self.weights.data_producer = iterable
            it_weights = self.weights.iterator(lag=0, return_trajindex=False, stride=self.stride, skip=self.skip,
                                               chunk=chunksize)
            if it_weights.number_of_trajectories() != iterable.number_of_trajectories():
                raise ValueError("number of weight arrays did not match number of input data sets. {} vs. {}"
                                 .format(it_weights.number_of_trajectories(), iterable.number_of_trajectories()))
        else:
            # if we only have a scalar, repeat it.
            import itertools
            it_weights = itertools.repeat(self.weights)

        # TODO: we could possibly optimize the case lag>0 and c0t=False using skip.
        # Access how much iterator hassle this would be.
        #self.skipped=0
        pg = ProgressReporter()
        pg.register(it.n_chunks, 'calculate covariances', stage=0)
        with it, pg.context(stage=0):
            self._init_covar(partial_fit, it.n_chunks)
            for data, weight in zip(it, it_weights):
                if self.lag != 0:
                    X, Y = data
                else:
                    X, Y = data, None

                if weight is not None:
                    if isinstance(weight, np.ndarray):
                        weight = weight.squeeze()[:len(X)]
                        # TODO: if the weight is exactly zero it makes not sense to add the chunk to running moments.
                        # however doing so, leads to wrong results...
                        # if np.all(np.abs(weight) < np.finfo(np.float).eps):
                        #     #print("skip")
                        #     self.skipped += len(X)
                        #     continue
                if self.remove_constant_mean is not None:
                    X = X - self.remove_constant_mean[np.newaxis, :]
                    if Y is not None:
                        Y = Y - self.remove_constant_mean[np.newaxis, :]

                try:
                    self._rc.add(X, Y, weights=weight)
                except MemoryError:
                    raise MemoryError('Covariance matrix does not fit into memory. '
                                      'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))
                pg.update(1, stage=0)

        if partial_fit:
            if '_rc' not in self.__serialize_fields:
                self.__serialize_fields.append('_rc')
        else:
            if '_rc' in self.__serialize_fields:
                self.__serialize_fields.remove('_rc')
        return self

    def partial_fit(self, X):
        """ incrementally update the estimates

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.
        """
        from pyemma.coordinates import source

        self._estimate(source(X), partial_fit=True)
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
    @deprecated('Please use the attribute "C00_".')
    def cov(self):
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    def C00_(self):
        """ Instantaneous covariance matrix """
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    @deprecated('Please use the attribute "C0t_".')
    def cov_tau(self):
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def C0t_(self):
        """ Time-lagged covariance matrix """
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def Ctt_(self):
        """ Covariance matrix of the time shifted data"""
        self._check_estimated()
        return self._rc.cov_YY(bessel=self.bessel)

    @property
    def nsave(self):
        if self.c00:
            return self._rc.storage_XX.nsave
        elif self.c0t:
            return self._rc.storage_XY.nsave

    @nsave.setter
    def nsave(self, ns):
        # potential bug? set nsave between partial fits?
        if self.c00:
            if self._rc.storage_XX.nsave <= ns:
                self._rc.storage_XX.nsave = ns
        if self.c0t:
            if self._rc.storage_XY.nsave <= ns:
                self._rc.storage_XY.nsave = ns
        if self.ctt:
            if self._rc.storage_YY.nsave <= ns:
                self._rc.storage_YY.nsave = ns

    @property
    def column_selection(self):
        return self._column_selection

    @column_selection.setter
    def column_selection(self, s):
        self._column_selection = s
        try:
            if self._rc is not None:
                self._rc.column_selection = s
        except AttributeError:
            pass
        self._estimated = False
        self.logger.debug('Modified column selection: estimate() needed for this change to take effect')
