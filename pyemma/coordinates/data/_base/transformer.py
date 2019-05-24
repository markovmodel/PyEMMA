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



from abc import ABCMeta, abstractmethod

import numpy as np


from pyemma._ext.sklearn.base import TransformerMixin
from pyemma.coordinates.data._base.datasource import DataSource, EncapsulatedIterator
from pyemma.coordinates.data._base.random_accessible import RandomAccessStrategy
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator

__all__ = ['Transformer', 'StreamingTransformer',
           'StreamingEstimationTransformer',  'StreamingTransformerRandomAccessStrategy',
           ]
__author__ = 'noe, marscher'


class Transformer(TransformerMixin, metaclass=ABCMeta):
    """ A transformer takes data and transforms it """

    @abstractmethod
    def describe(self):
        r""" Get a descriptive string representation of this class."""
        raise NotImplementedError()

    def transform(self, X):
        r"""Maps the input data through the transformer to correspondingly
        shaped output data array/list.

        Parameters
        ----------
        X : ndarray(T, n) or list of ndarray(T_i, n)
            The input data, where T is the number of time steps and n is the
            number of dimensions.
            If a list is provided, the number of time steps is allowed to vary,
            but the number of dimensions are required to be to be consistent.

        Returns
        -------
        Y : ndarray(T, d) or list of ndarray(T_i, d)
            The mapped data, where T is the number of time steps of the input
            data and d is the output dimension of this transformer. If called
            with a list of trajectories, Y will also be a corresponding list of
            trajectories
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                mapped = self._transform_array(X)
                return mapped
            else:
                raise TypeError('Input has the wrong shape: %s with %i'
                                ' dimensions. Expecting a matrix (2 dimensions)'
                                % (str(X.shape), X.ndim))
        elif isinstance(X, (list, tuple)):
            out = []
            for x in X:
                mapped = self._transform_array(x)
                out.append(mapped)
            return out
        else:
            raise TypeError('Input has the wrong type: %s '
                            '. Either accepting numpy arrays of dimension 2 '
                            'or lists of such arrays' % (str(type(X))))

    @abstractmethod
    def _transform_array(self, X):
        r"""
        Initializes the parametrization.

        Parameters
        ----------
        X : ndarray(T, n)
            The input data, where T is the number of time steps and
            n is the number of dimensions.

        Returns
        -------
        Y : ndarray(T, d)
            The projected data, where T is the number of time steps of the
            input data and d is the output dimension of this transformer.

        """
        raise NotImplementedError()


class StreamingTransformer(Transformer, DataSource):

    r""" Basis class for pipelined Transformers.

    This class derives from DataSource, so follow up pipeline elements can stream
    the output of this class.

    Parameters
    ----------
    chunksize : int (optional)
        the chunksize used to batch process underlying data.

    """
    def __init__(self, chunksize=None):
        super(StreamingTransformer, self).__init__(chunksize=chunksize)
        self.data_producer = None
        self._estimated = True  # this class should only transform data and need no estimation.

    @abstractmethod
    def dimension(self):
        raise NotImplementedError()

    @property
    # overload of DataSource
    def data_producer(self):
        if not hasattr(self, '_data_producer'):
            return None
        return self._data_producer

    @data_producer.setter
    def data_producer(self, dp):
        self._data_producer = dp
        if dp is not None and not isinstance(dp, DataSource):
            raise ValueError('can not set data_producer to non-iterable class of type {}'.format(type(dp)))
        # register random access strategies
        self._set_random_access_strategies()

    def _set_random_access_strategies(self):
        if self.in_memory and self._Y_source is not None:
            self._ra_cuboid = self._Y_source._ra_cuboid
            self._ra_linear_strategy = self._Y_source._ra_linear_strategy
            self._ra_linear_itraj_strategy = self._Y_source._ra_linear_itraj_strategy
            self._ra_jagged = self._Y_source._ra_jagged
            self._is_random_accessible = True
        elif self.data_producer is not None:
            self._ra_jagged = \
                StreamingTransformerRandomAccessStrategy(self, self.data_producer._ra_jagged)
            self._ra_linear_itraj_strategy = \
                StreamingTransformerRandomAccessStrategy(self, self.data_producer._ra_linear_itraj_strategy)
            self._ra_linear_strategy = \
                StreamingTransformerRandomAccessStrategy(self, self.data_producer._ra_linear_strategy)
            self._ra_cuboid = \
                StreamingTransformerRandomAccessStrategy(self, self.data_producer._ra_cuboid)
            self._is_random_accessible = self.data_producer._is_random_accessible
        else:
            self._ra_jagged = self._ra_linear_itraj_strategy = self._ra_linear_strategy \
                = self._ra_cuboid = None
            self._is_random_accessible = False

    def _map_to_memory(self, stride=1):
        super(StreamingTransformer, self)._map_to_memory(stride)
        self._set_random_access_strategies()

    def _clear_in_memory(self):
        super(StreamingTransformer, self)._clear_in_memory()
        self._set_random_access_strategies()

    def _create_iterator(self, skip=0, chunk=None, stride=1, return_trajindex=True, cols=None):
        real_iter = self.data_producer.iterator(
            skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex, cols=cols
        )
        return EncapsulatedIterator(self, iterator=real_iter, transform_function=self._transform_array,
                                    skip=skip, chunk=chunk, stride=stride,
                                    return_trajindex=return_trajindex, cols=cols)

    @property
    def chunksize(self):
        """chunksize defines how much data is being processed at once."""
        if not self.data_producer:
            return self.default_chunksize
        return self.data_producer.chunksize

    @chunksize.setter
    def chunksize(self, value):
        if self.data_producer is None:
            if not isinstance(value, (type(None), int)):
                raise ValueError('chunksize has to be of type: None or int')
            if isinstance(value, int) and value < 0:
                raise ValueError("Chunksize of %s was provided, but has to be >= 0" % value)
            self._default_chunksize = value
        else:
            self.data_producer.chunksize = value

    def number_of_trajectories(self, stride=1):
        return self.data_producer.number_of_trajectories(stride)

    def trajectory_length(self, itraj, stride=1, skip=0):
        return self.data_producer.trajectory_length(itraj, stride=stride, skip=skip)

    def trajectory_lengths(self, stride=1, skip=0):
        return self.data_producer.trajectory_lengths(stride=stride, skip=skip)

    def n_frames_total(self, stride=1, skip=0):
        return self.data_producer.n_frames_total(stride=stride, skip=skip)


class StreamingEstimationTransformer(StreamingTransformer, StreamingEstimator):
    def __init__(self):
        super(StreamingEstimationTransformer, self).__init__()
        self._estimated = False

    """ Basis class for pipelined Transformers, which perform also estimation. """
    def estimate(self, X, **kwargs):
        super(StreamingEstimationTransformer, self).estimate(X, **kwargs)
        # we perform the mapping to memory exactly here, because a StreamingEstimator on its own
        # has not output to be mapped. Only the combination of Estimation/Transforming has this feature.
        if self.in_memory and not self._mapping_to_mem_active:
            self._map_to_memory()
        return self

    def get_output(self, dimensions=slice(0, None), stride=1, skip=0, chunk=None):
        if not self._estimated:
            self.estimate(self.data_producer, stride=stride)

        return super(StreamingTransformer, self).get_output(dimensions, stride, skip, chunk)


class StreamingTransformerRandomAccessStrategy(RandomAccessStrategy):
    def __init__(self, source, parent_strategy):
        super(StreamingTransformerRandomAccessStrategy, self).__init__(source)
        self._parent_strategy = parent_strategy
        self._max_slice_dimension = self._parent_strategy._max_slice_dimension

    def _handle_slice(self, idx):
        dimension_slice = slice(None, None, None)
        if len(idx) == self.max_slice_dimension:
            # a dimension slice was passed
            idx, dimension_slice = idx[0:self.max_slice_dimension-1], idx[-1]
        X = self._parent_strategy[idx]
        if isinstance(X, list):
            return [self._source._transform_array(Y)[:, dimension_slice].astype(self._source.output_type()) for Y in X]
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                return self._source._transform_array(X)[:, dimension_slice].astype(self._source.output_type())
            elif X.ndim == 3:
                dims = self._get_indices(dimension_slice, self._source.ndim)
                ndims = len(dims)
                old_shape = X.shape
                new_shape = (X.shape[0], X.shape[1], ndims)
                mapped_data = np.empty(new_shape, dtype=self._source.output_type())
                for i in range(old_shape[0]):
                    mapped_data[i] = self._source._transform_array(X[i])[:, dims]
                return mapped_data

        else:
            raise IndexError("Could not handle object of type %s for transformer slicing" % str(type(X)))
