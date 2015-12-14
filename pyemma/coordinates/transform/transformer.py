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

import numpy as np
import six
from six.moves import range

from pyemma._base.estimator import Estimator
from pyemma._base.logging import Loggable
from pyemma.coordinates.data import DataInMemory
from pyemma.coordinates.data.datasource import DataSource
from pyemma.coordinates.data.datasource import DataSourceIterator
from pyemma.coordinates.data.iterable import Iterable
from pyemma.util import types as _types
from pyemma.util.annotators import deprecated
from pyemma.util.exceptions import NotConvergedWarning

__all__ = ['Transformer']
__author__ = 'noe, marscher'


def _to_data_producer(X):
    from pyemma.coordinates.data.data_in_memory import DataInMemory as _DataInMemory
    # this is a pipelining stage, so let's parametrize from it
    if isinstance(X, Transformer):
        inputstage = X
    # second option: data is array or list of arrays
    else:
        data = _types.ensure_traj_list(X)
        inputstage = _DataInMemory(data)

    return inputstage


class Transformer(six.with_metaclass(ABCMeta, DataSource, Estimator, Loggable)):
    r""" Basis class for pipeline objects

    Parameters
    ----------
    chunksize : int (optional)
        the chunksize used to batch process underlying data

    """

    def __init__(self, chunksize=0):
        super(Transformer, self).__init__(chunksize)
        if chunksize is not None:
            self._logger.warning("Given deprecated argument 'chunksize=%s'"
                                 " to transformer. Ignored - please set the "
                                 "chunksize in the reader" % chunksize)
        self._data_producer = None
        # todo ??
        self._parametrized = False
        self._param_with_stride = 1
        # allow children of this class to implement their own progressbar handling
        self._custom_param_progress_handling = False

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True):
        return TransformerIterator(self, skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex)

    @property
    def data_producer(self):
        r"""where the transformer obtains its data."""
        return self._data_producer

    @data_producer.setter
    def data_producer(self, dp):
        if dp is not self._data_producer:
            self._logger.debug("reset (previous) parametrization state, since"
                               " data producer has been changed.")
            self._parametrized = False
        self._data_producer = dp

    #@Iterable.chunksize
    def chunksize(self):
        """chunksize defines how much data is being processed at once."""
        return self.data_producer.chunksize

    @Iterable.chunksize.setter
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

    @abstractmethod
    def describe(self):
        r""" Get a descriptive string representation of this class."""
        pass

    def fit(self, X, **kwargs):
        r"""For compatibility with sklearn"""
        self.data_producer = _to_data_producer(X)
        self.estimate(X, **kwargs)
        return self

    def fit_transform(self, X, **kwargs):
        r"""For compatibility with sklearn"""
        self.fit(X, **kwargs)
        return self.transform(X)

    def estimate(self, X, **kwargs):
        if not isinstance(X, Iterable):
            if isinstance(X, np.ndarray):
                X = DataInMemory(X, self.chunksize)
            else:
                raise ValueError("no")

        model = None
        try:
            # start
            self._param_init()
            model = super(Transformer, self).estimate(X, **kwargs)
        except NotConvergedWarning as ncw:
            self._logger.info("Presumely finished estimation. Message: %s" % ncw)
        # finish
        self._param_finish()
        # memory mode? Then map all results. Avoid recursion here, if parametrization
        # is triggered from get_output
        if self.in_memory and not self._mapping_to_mem_active:
            self._map_to_memory()

        # finish parametrization
        #if not self._custom_param_progress_handling:
        #    self._progress_force_finish(0)

        self._estimated = True

        return model

    @deprecated("use estimate")
    def parametrize(self):
        if self._data_producer is None:
            raise RuntimeError("This estimator has no data source given, giving up.")

        return self.estimate(self.data_producer)

    @deprecated("use fit.")
    def transform(self, X):
        r"""Maps the input data through the transformer to correspondingly shaped output data array/list.

        Parameters
        ----------
        X : ndarray(T, n) or list of ndarray(T_i, n)
            The input data, where T is the number of time steps and n is the number of dimensions.
            If a list is provided, the number of time steps is allowed to vary, but the number of dimensions are
            required to be to be consistent.

        Returns
        -------
        Y : ndarray(T, d) or list of ndarray(T_i, d)
            The mapped data, where T is the number of time steps of the input data and d is the output dimension
            of this transformer. If called with a list of trajectories, Y will also be a corresponding list of
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
            The input data, where T is the number of time steps and n is the number of dimensions.

        Returns
        -------
        Y : ndarray(T, d)
            The projected data, where T is the number of time steps of the input data and d is the output dimension
            of this transformer.

        """
        pass

    def _param_init(self):
        r"""
        Initializes the parametrization.
        """
        pass

    def _param_finish(self):
        r"""
        Finalizes the parametrization.
        """
        pass


class TransformerIterator(DataSourceIterator):

    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False):
        super(TransformerIterator, self).__init__(data_source, return_trajindex=return_trajindex)
        self._it = self._data_source.data_producer._create_iterator(
                skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex
        )
        self.context = self._it.context

    def _n_chunks(self, stride=None):
        return self._it._n_chunks(stride=stride)

    def close(self):
        self._it.close()

    def next_chunk(self):
        X = self._it.next_chunk()
        return self._data_source.transform(X)