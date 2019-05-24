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


import functools

import numpy as np

from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.coordinates.data._base.datasource import DataSourceIterator, DataSource
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator
from pyemma.coordinates.data.util.traj_info_cache import TrajInfo
from pyemma.util.annotators import fix_docs


@fix_docs
class NumPyFileReader(DataSource, SerializableMixIn):
    __serialize_version = 0

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
        self._is_reader = True

        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        for f in filenames:
            if not f.endswith('.npy'):
                raise ValueError('given file "%s" is not supported by this'
                                 ' reader, since it does not end with .npy' % f)

        self.mmap_mode = mmap_mode
        self.filenames = filenames

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        return NPYIterator(self, skip=skip, chunk=chunk, stride=stride,
                           return_trajindex=return_trajindex, cols=cols)

    def describe(self):
        shapes = [(x, self.ndim) for x in self.trajectory_lengths()]
        return "[NumpyFileReader arrays with shapes: {}]".format(shapes)

    def _reshape(self, array):
        """
        checks shapes, eg convert them (2d), raise if not possible
        after checks passed, set self._array and return it.
        """

        if array.ndim == 1:
            array = np.atleast_2d(array).T
        else:
            shape = array.shape
            # hold first dimension, multiply the rest
            shape_2d = (shape[0],
                        functools.reduce(lambda x, y: x * y, shape[1:]))
            array = np.reshape(array, shape_2d)
        return array

    def _load_file(self, itraj):
        filename = self._filenames[itraj]
        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            arr = self._reshape(x)
        else:
            raise ValueError("given file '%s' is not a NumPy array. Make sure"
                             " it has a .npy extension" % filename)
        return arr

    def _get_traj_info(self, filename):
        idx = self.filenames.index(filename)
        array = self._load_file(idx)
        length, ndim = np.shape(array)

        return TrajInfo(ndim, length)

    def __reduce__(self):
        # serialize only the constructor arguments.
        return NumPyFileReader, (self.filenames, self.chunksize, self.mmap_mode)


class NPYIterator(DataInMemoryIterator):

    def close(self):
        if hasattr(self, 'data') and self.data is not None:
            # delete the memmap to close it.
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.memmap.html
            del self.data
            self.data = None

    @DataInMemoryIterator._select_file_guard
    def _select_file(self, itraj):
        self.close()
        assert itraj < self.number_of_trajectories()
        self.data = self._data_source._load_file(itraj)

    def _next_chunk(self):
        return self._next_chunk_impl(self.data)
