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


from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import six

from pyemma._base.estimator import Estimator
from pyemma._ext.sklearn.base import TransformerMixin
from pyemma.coordinates.data import DataInMemory
from pyemma.coordinates.data._base.datasource import DataSource, DataSourceIterator
from pyemma.coordinates.data._base.iterable import Iterable
from pyemma.coordinates.util.change_notification import (inform_children_upon_change,
                                                         NotifyOnChangesMixIn)
from pyemma.util.exceptions import NotConvergedWarning
from six.moves import range
import numpy as np


__all__ = ['Transformer', 'StreamingTransformer']
__author__ = 'noe, marscher'


class Transformer(six.with_metaclass(ABCMeta, Estimator, TransformerMixin)):
    """ A transformer takes data and transforms it """

    @abstractmethod
    def describe(self):
        r""" Get a descriptive string representation of this class."""
        pass

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
        pass


class StreamingTransformer(Transformer, DataSource, NotifyOnChangesMixIn):

    r""" Basis class for pipelined Transformers

    Parameters
    ----------
    chunksize : int (optional)
        the chunksize used to batch process underlying data

    """

    def __init__(self, chunksize=1000):
        super(StreamingTransformer, self).__init__(chunksize)
        self._estimated = False
        self._data_producer = None

    @property
    # overload of DataSource
    def data_producer(self):
        return self._data_producer

    @data_producer.setter
    @inform_children_upon_change
    def data_producer(self, dp):
        if dp is not self._data_producer:
            # first unregister from current dataproducer
            if self._data_producer is not None and isinstance(self._data_producer, NotifyOnChangesMixIn):
                self._data_producer._stream_unregister_child(self)
            # then register this instance as a child of the new one.
            if dp is not None and isinstance(dp, NotifyOnChangesMixIn):
                dp._stream_register_child(self)
        self._data_producer = dp

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True):
        return StreamingTransformerIterator(self, skip=skip, chunk=chunk, stride=stride,
                                            return_trajindex=return_trajindex)

    def estimate(self, X, **kwargs):
        # TODO: X is either Iterable of an array
        if not isinstance(X, Iterable):
            if isinstance(X, np.ndarray):
                X = DataInMemory(X, self.chunksize)
                self.data_producer = X
            else:
                raise ValueError("no array given")

        model = None
        # run estimation
        try:
            model = super(StreamingTransformer, self).estimate(X, **kwargs)
        except NotConvergedWarning as ncw:
            self._logger.info(
                "Presumely finished estimation. Message: %s" % ncw)
        # memory mode? Then map all results. Avoid recursion here, if parametrization
        # is triggered from get_output
        if self.in_memory and not self._mapping_to_mem_active:
            self._map_to_memory()

        self._estimated = True

        return model

    def get_output(self, dimensions=slice(0, None), stride=1, skip=0, chunk=0):
        if not self._estimated:
            self.estimate(self.data_producer, stride=stride)

        return super(StreamingTransformer, self).get_output(dimensions, stride, skip, chunk)

    #@deprecated("Please use estimate")
    def parametrize(self, stride=1):
        if self._data_producer is None:
            raise RuntimeError(
                "This estimator has no data source given, giving up.")

        return self.estimate(self.data_producer, stride=stride)

    # get lengths etc. info from parent

    @DataSource.chunksize.getter
    def chunksize(self):
        """chunksize defines how much data is being processed at once."""
        return self.data_producer.chunksize

    @DataSource.chunksize.setter
    def chunksize(self, size):
        if not size >= 0:
            raise ValueError("chunksize has to be positive")

        self.data_producer.chunksize = int(size)

    def number_of_trajectories(self):
        r"""
        Returns the number of trajectories.

        Returns
        -------
            int : number of trajectories
        """
        return self.data_producer.number_of_trajectories()

    def trajectory_length(self, itraj, stride=1, skip=None):
        r"""
        Returns the length of trajectory of the requested index.

        Parameters
        ----------
        itraj : int
            trajectory index
        stride : int
            return value is the number of frames in the trajectory when
            running through it with a step size of `stride`.

        Returns
        -------
        int : length of trajectory
        """
        return self.data_producer.trajectory_length(itraj, stride=stride, skip=skip)

    def trajectory_lengths(self, stride=1, skip=0):
        r"""
        Returns the length of each trajectory.

        Parameters
        ----------
        stride : int
            return value is the number of frames of the trajectories when
            running through them with a step size of `stride`.
        skip : int
            skip parameter

        Returns
        -------
        array(dtype=int) : containing length of each trajectory
        """
        return self.data_producer.trajectory_lengths(stride=stride, skip=skip)

    def n_frames_total(self, stride=1):
        r"""
        Returns total number of frames.

        Parameters
        ----------
        stride : int
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`.

        Returns
        -------
        int : n_frames_total
        """
        return self.data_producer.n_frames_total(stride=stride)


class StreamingTransformerIterator(DataSourceIterator):

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        super(StreamingTransformerIterator, self).__init__(
            data_source, return_trajindex=return_trajindex)
        self._it = self._data_source.data_producer._create_iterator(
            skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex
        )
        self.state = self._it.state

    def close(self):
        self._it.close()

    def _next_chunk(self):
        X = self._it._next_chunk()
        return self._data_source._transform_array(X)
