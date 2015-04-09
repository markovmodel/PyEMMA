'''
Created on 07.04.2015

@author: marscher
'''
import numpy as np

from pyemma.coordinates.transform.transformer import Transformer
from pandas.io.parsers import TextFileReader
import functools


class ReaderInterface(Transformer):
    """basic interface for readers
    """

    def __init__(self, chunksize=100):
        super(ReaderInterface, self).__init__(chunksize=chunksize)
        # TODO: think about if this should be none or self
        self.data_producer = None

        # internal counters
        self._t = 0
        self._itraj = 0

        # lengths and dims
        self._ntraj = -1
        self._ndim = -1
        self._lengths = []

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self._ndim

    def trajectory_length(self, itraj, stride=1):
        return (self._lengths[itraj] - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1):
        return [(l - 1) // stride + 1 for l in self._lengths]

    def number_of_trajectories(self):
        return self._ntraj

    def n_frames_total(self, stride=1):
        # FIXME: in case of 1 traj, this returns 1!!!
        if stride == 1:
            self._logger.debug("self._lengths= %s " % self._lengths)
            return np.sum(self._lengths)
        else:
            return sum(self.trajectory_lengths(stride))

    def _add_array_to_storage(self, array):
        # checks shapes, eg convert them (2d), raise if not possible
        # after checks passed, add array to self._data

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

        #self._logger.debug("added array with shape %s" % str(array.shape))
        self._data.append(array)


class NumPyFileReader(ReaderInterface):

    """reads NumPy files in chunks. Supports .npy and .npz files

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
            if not (f.endswith('.npy') or f.endswith('.npz')):
                raise ValueError('given file "%s" is not supported'
                                 ' by this reader' % f)

        self.mmap_mode = mmap_mode

        # current storage, holds mmapped arrays
        self._data = []

        # current file handle
        self._fh = None

        self.__set_dimensions_and_lenghts()

        self._parametrized = True

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        if self._fh is not None:
            self._fh.close()

    def __load_file(self, filename):
        assert filename in self._filenames

        if self._fh is not None:
            # name already open?
            if self._fh.name == filename:
                return
            else:
                self._fh.close()

        self._logger.debug("opening file %s" % filename)
        self._fh = open(filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            self._add_array_to_storage(x)

        # in this case the file might contain several arrays
        elif filename.endswith('.npz'):
            # closes file handle
            npz_file = np.load(self._fh, mmap_mode=self.mmap_mode)
            for _, arr in npz_file.items():
                self._add_array_to_storage(arr)
        else:
            raise ValueError("given file '%s' is not a NumPy array. Make sure it has"
                             " either an .npy or .npz extension" % filename)

    def __set_dimensions_and_lenghts(self):
        for f in self._filenames:
            self.__load_file(f)

        self._lengths += [np.shape(x)[0] for x in self._data]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self._data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

        self._ntraj = len(self._data)

    def _next_chunk(self, lag=0, stride=1):

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self._filenames) - 1):
            # close file handles and open new ones
            self._t = 0
            self._itraj += 1
            self.__load_file(self._filenames[self._itraj])
            # we open self._mditer2 only if requested due lag parameter!
            self._curr_lag = 0

        traj_len = self.trajectory_length(self._itraj, stride)
        traj = self._data[self._itraj]

        # complete trajectory mode
        if self._chunksize == 0:
            X = traj[::stride]
            self._itraj += 1

            if lag == 0:
                return X
            else:
                Y = traj[lag * stride:traj_len:stride]
                return (X, Y)
        # chunked mode
        else:
            upper_bound = min(self._t + self._chunksize * stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)

            X = traj[slice_x]
            last_t = self._t
            self._t += X.shape[0]

            if self._t >= traj_len:
                self._itraj += 1
                self._t = 0

            if lag == 0:
                return X
            else:
                # its okay to return empty chunks
                upper_bound = min(
                    last_t + (lag + self._chunksize) * stride, traj_len)
                slice_y = slice(last_t + lag, upper_bound, stride)

                Y = traj[slice_y]
                return X, Y


class CSVReader(ReaderInterface):
    """reads tabulated data from given files in chunked mode

    Parameters
    ----------
    filenames : list of strings
        filenames (including paths) to read
    sep : str
        separator to use during parsing the file
    chunksize : int 
        how many lines to process at once

    """

    def __init__(self, filenames, sep=' ', chunksize=1000):
        super(CSVReader, self).__init__(chunksize=chunksize)
        self.data_producer = self

        if not isinstance(filenames, (tuple, list)):
            filenames = [filenames]
        self._filenames = filenames

        self.sep = sep

        self._t = 0
        self._itraj = 0

        self._ndim = 0
        self._ntraj = 0
        self._lengths = []

        self._reader = None
        self.__set_dimensions_and_lenghts()

        # lagged access
        self._lagged_reader = None
        self._current_lag = 0

        self._parametrized = True

    def __set_dimensions_and_lenghts(self):
        # number of trajectories/data sets
        self._ntraj = len(self._filenames)
        if self._ntraj == 0:
            raise ValueError("no valid data")

        ndims = []
        for f in self._filenames:
            try:
                # determine file length
                with open(f) as fh:
                    self._lengths.append(sum(1 for _ in fh))
                    fh.seek(0)
                    line = fh.readline()
                    dim = np.fromstring(line, sep=self.sep).shape[0]
                    ndims.append(dim)

            # parent of IOError, OSError *and* WindowsError where available
            except EnvironmentError:
                self._logger.exception()
                self._logger.error("removing %s from list" % f)
                self._filenames.remove(f)

        # check all files have same dimensions
        if not len(np.unique(ndims)) == 1:
            raise ValueError("input files have different dims")
        else:
            self._ndim = dim

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0

    def _open_file(self):
        fn = self._filenames[self._itraj]

        # do not open same file
        # if self._reader and self._reader.f == fn:
        #    return
        self._reader = TextFileReader(fn, chunksize=self.chunksize)

    def _open_file_lagged(self, skip):
        self._logger.debug("opening lagged file with skip=%i" % skip)
        fn = self._filenames[self._itraj]

        # do not open same file, if we can still read something
        if self._lagged_reader and self._lagged_reader.f == fn:
            return

        self._lagged_reader = TextFileReader(
            fn, chunksize=self.chunksize, skip=skip)

    def _next_chunk(self, lag=0, stride=1):

        self._open_file()

        if (self._t >= self.trajectory_length(self._itraj, stride=stride) and
                self._itraj < len(self._filenames) - 1):
            # close file handles and open new ones
            self._t = 0
            self._itraj += 1
#            self.__load_file(self._filenames[self._itraj])
#         if self._t >= self.trajectory_length(self._itraj):
#             self._logger.info("t(%i) >= tlen(%i)" %
#                               (self._t, self.trajectory_length(self._itraj)))
#             self._t = 0
#             self._itraj += 1
            self._open_file()
            self._open_file_lagged()

        if lag != self._current_lag:
            self._current_lag = lag
            self._open_file_lagged(skip=lag)

        if lag == 0:
            X = self._reader.get_chunk().values
            self._t += X.shape[0]
            return X
        else:
            X = self._reader.get_chunk().values
            self._t += X.shape[0]
            assert self._lagged_reader
            try:
                Y = self._lagged_reader.get_chunk().values
            except StopIteration:
                Y = np.empty(0)
            return X, Y
