# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Created on 07.04.2015

@author: marscher
'''
import numpy as np
import functools

from pyemma.coordinates.data.interface import ReaderInterface
from pyemma.util.progressbar._impl import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar


class NumPyFileReader(ReaderInterface):

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

        self._parametrized = True

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        self._close()

    def describe(self):
        return "[NumpyFileReader arrays with shape %s]" % [np.shape(x)
                                                           for x in self._data]

    def __reshape(self, array):
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

    def __load_file(self, filename):
        self._close()
        self._logger.debug("opening file %s" % filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            arr = self.__reshape(x)
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
        pg = ProgressBar(n, description="get lengths/dim")
        pg.eta_every = 1

        for f in self._filenames:
            array = self.__load_file(f)
            self._lengths.append(np.shape(array)[0])
            ndims.append(np.shape(array)[1])
            self._close()
            pg.numerator += 1
            show_progressbar(pg)

        # ensure all trajs have same dim
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

        self._ntraj = len(self._filenames)

    def _next_chunk(self, context=None):

        # if no file is open currently, open current index.
        if self._array is None:
            self.__load_file(self._filenames[self._itraj])

        traj_len = self._lengths[self._itraj]
        traj = self._array

        # if stride by dict, update traj length accordingly
        if not context.uniform_stride:
            traj_len = context.ra_trajectory_length(self._itraj)

        # complete trajectory mode
        if self._chunksize == 0:
            if not context.uniform_stride:
                X = traj[context.ra_indices_for_traj(self._itraj)]
                self._itraj += 1

                # skip the trajs that are not in the stride dict
                while self._itraj < self.number_of_trajectories() \
                        and (self._itraj not in context.traj_keys):
                    self._itraj += 1
                self._array = None
            else:
                X = traj[::context.stride]
                self._itraj += 1

            if context.lag == 0:
                return X
            else:
                if not context.uniform_stride:
                    raise ValueError("Requested lagged data but was in random access mode. This is not supported.")
                else:
                    Y = traj[context.lag::context.stride]
                return X, Y

        # chunked mode
        else:
            if not context.uniform_stride:
                X = traj[context.ra_indices_for_traj(self._itraj)[self._t:min(self._t + self.chunksize, traj_len)]]
                upper_bound = min(self._t + self.chunksize, traj_len)
            else:
                upper_bound = min(self._t + self._chunksize * context.stride, traj_len)
                slice_x = slice(self._t, upper_bound, context.stride)
                X = traj[slice_x]

            if context.lag != 0:
                upper_bound_Y = min(self._t + context.lag + self._chunksize * context.stride, traj_len)
                slice_y = slice(self._t + context.lag, upper_bound_Y, context.stride)
                Y = traj[slice_y]

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
                    self.__load_file(self._filenames[self._itraj])

            if context.lag == 0:
                return X
            else:
                return X, Y
