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
Created on 07.04.2015

@author: marscher
'''

from __future__ import absolute_import

import functools

from pyemma._base.progress import ProgressReporter

import numpy as np
from pyemma.coordinates.data.datasource import DataSource, DataSourceIterator


class NumPyFileReader(DataSource, ProgressReporter):

    """reads NumPy files in chunks. Supports .npy files

    Parameters
    ----------
    filenames : str or list of strings

    chunksize : int
        how many rows are read at once

    mmap_mode : str (optional), default='r'
        binary NumPy arrays are being memory mapped using this flag.
    """

    def __init__(self, filenames, chunksize=1000, mmap_mode='r'):
        super(NumPyFileReader, self).__init__(chunksize=chunksize)

        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        self._filenames = filenames

        for f in self._filenames:
            if not f.endswith('.npy'):
                raise ValueError('given file "%s" is not supported by this'
                                 ' reader, since it does not end with .npy' % f)

        self.mmap_mode = mmap_mode

        # currently opened array
        self._array = None

        self.__set_dimensions_and_lenghts()
        
    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=False):
        return NPYIterator(self, skip, chunk, stride, return_trajindex)

    def describe(self):
        return "[NumpyFileReader arrays with shape %s]" % [np.shape(x)
                                                           for x in self._data]

    def _reshape(self, array):
        """
        checks shapes, eg convert them (2d), raise if not possible
        after checks passed, set self._array and return it.
        """

        if array.ndim == 1:
            array = np.atleast_2d(array).T
        elif array.ndim == 2:
            pass
        else:
            shape = array.shape
            # hold first dimension, multiply the rest
            shape_2d = (shape[0],
                        functools.reduce(lambda x, y: x * y, shape[1:]))
            array = np.reshape(array, shape_2d)
        return array

    def _load_file(self, itraj):
        self._close()
        filename = self._filenames[itraj]
        self._logger.debug("opening file %s" % filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            arr = self._reshape(x)
        else:
            raise ValueError("given file '%s' is not a NumPy array. Make sure"
                             " it has a .npy extension" % filename)
        self._array = arr
        return arr

    def _close(self):
        if self._array is None:
            return

        if __debug__:
            self._logger.debug("delete filehandle")
        del self._array
        self._array = None

    def __set_dimensions_and_lenghts(self):
        ndims = []
        n = len(self._filenames)
        self._progress_register(n, description="get lengths/dim")

        for ii, f in enumerate(self._filenames):
            array = self._load_file(ii)
            self._lengths.append(np.shape(array)[0])
            ndims.append(np.shape(array)[1])
            self._close()
            self._progress_update(1)

        # ensure all trajs have same dim
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

        self._ntraj = len(self._filenames)


class NPYIterator(DataSourceIterator):
    
    def close(self):
        self._data_source._close()
        raise StopIteration()

    def next_chunk(self):
        if self.current_trajindex >= self._data_source.ntraj:
            self.close()

        # TODO: :refactor
        context = self

        # if no file is open currently, open current index.
        if self._array is None:
            self._array = self._data_source._load_file(self._itraj)

        traj_len = len(self._array)
        traj = self._array

        # skip only if complete trajectory mode or first chunk
        skip = self.skip if self.chunksize == 0 or self._t == 0 else 0

        # if stride by dict, update traj length accordingly
        if not context.uniform_stride:
            traj_len = context.ra_trajectory_length(self._itraj)

        # complete trajectory mode
        if self.chunksize == 0:
            if not context.uniform_stride:
                X = traj[context.ra_indices_for_traj(self._itraj)]
                self._itraj += 1

                # skip the trajs that are not in the stride dict
                while self._itraj < self.number_of_trajectories() \
                        and (self._itraj not in context.traj_keys):
                    self._itraj += 1
                self._array = None
            else:
                X = traj[skip::context.stride]
                self._itraj += 1

            return X

        # chunked mode
        else:
            if not context.uniform_stride:
                X = traj[context.ra_indices_for_traj(self._itraj)[self._t:min(self._t + self.chunksize, traj_len)]]
                upper_bound = min(self._t + self.chunksize, traj_len)
            else:
                upper_bound = min(skip + self._t + self.chunksize * context.stride, traj_len)
                slice_x = slice(skip + self._t, upper_bound, context.stride)
                X = traj[slice_x]

            # set new time position
            self._t = upper_bound

            if self._t >= traj_len:
                if __debug__:
                    self._logger.debug("reached bounds of array, open next.")
                self._itraj += 1
                self._t = 0

                # if we have a dictionary, skip trajectories that are not in the key set
                while not context.uniform_stride and self._itraj < self.number_of_trajectories() \
                        and (self._itraj not in context.traj_keys):
                    self._itraj += 1

                # if time index scope ran out of len of current trajectory, open next file.
                if self._itraj <= self.number_of_trajectories() - 1:
                    self._data_source._load_file(self._itraj)

            return X
