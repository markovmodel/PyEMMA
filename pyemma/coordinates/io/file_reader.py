'''
Created on 07.04.2015

@author: marscher
'''
import numpy as np

from pyemma.coordinates.transform.transformer import Transformer


class FileReader(Transformer):

    """
    Parameters
    ----------
    filenames : str or list of strings

    chunksize : int
        how many rows are read at once

    mmap_mode : str (optional), default='r'
        binary NumPy arrays are being memory mapped using this flag.
    """

    def __init__(self, filenames, chunksize=1000, mmap_mode='r'):
        Transformer.__init__(self, chunksize=chunksize)
        self.data_producer = self

        self._filenames = filenames
        self.mmap_mode = mmap_mode

        # current storage, is used in case of npz arrays
        self._data = []

        # current file handle
        self._fh = None

        # internal counters
        self._t = 0
        self._itraj = 0
        self._ntraj = len(self._filenames)

        self._ndim = 0
        self._lengths = []

        self._parametrized = True

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

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        if self._fh is not None:
            self._fh.close()

        self._current_file_is_ascii = False
        self._current_file_is_ascii = 0

    def __open_file(self, i):
        assert i <= self._ntraj

        if self._fh is not None:
            # name already open?
            if self._fh.name == self._filenames[self._itraj]:
                return
            else:
                self._fh.close()

        # handle all kinds of types....
        filename = self._filenames[i]
        self._logger.debug("opening file %s" % filename)
        self._fh = open(filename)

        if filename.endswith('.npy'):
            x = np.load(filename, mmap_mode=self.mmap_mode)
            self._data = [x]

        # in this case the file might contain several arrays
        elif filename.endswith('.npz'):
            # closes file handle
            npz_file = np.load(self._fh, mmap_mode=self.mmap_mode)
            self._data = npz_file.items()
        else:
            raise ValueError("unsupported file %s" % filename)

    def __set_dimensions_and_lenghts(self):
        for ii, _ in enumerate(self._filenames):
            self.__open_file(ii)

            self._logger.debug(
                "first dimension of current data = %s" % str(np.shape(self._data)[0]))
            if self._current_file_is_ascii:
                length = 0
                first_line = True
                # we have to read this
                for line in self._fh:
                    if first_line:
                        arr = np.fromstring(line)
                        dim = arr.shape[0]
                    length += dim
            else:
                self._lengths += [np.shape(self._data)[0]]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self._data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

    def next_chunk(self, lag=0, stride=1):

        self.__open_file(self._itraj)

        if self._t >= self._ntraj:
            self.reset(stride)

        # have mmem mapped arrays
        if not self.__current_file_is_ascii:
            traj_len = self._lengths[self._itraj]
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


class CSVReader(Transformer):

    """ a stupid csv reader
    """

    def __init__(self, ascii_files, chunksize=100):
        Transformer.__init__(self, chunksize=chunksize)
        self.data_producer = self

        # todo check list of files
        # TODO: ensure no .npy|z files are in this list
        if not isinstance(ascii_files, (list, tuple)):
            ascii_files = [ascii_files]
        self._filenames = ascii_files
        self._ntraj = len(self._filenames)

        self._fh = None
        self._itraj = 0
        self._t = 0
        self._lengths = []
        self._ndim = 0

        # ascii files
        self._current_file_is_ascii = False
        self._ascii_pos = 0

        self.__set_lengths_and_dimension()

    def __open_file(self, i):
        assert i <= self._ntraj

        if self._fh is not None:
            # name already open?
            if self._fh.name == self._filenames[self._itraj]:
                return
            else:
                self._logger.debug("closing file %s" %self._fh.name)
                self._fh.close()

        # handle all kinds of types....
        filename = self._filenames[i]
        self._logger.debug("opening file %s" % filename)
        self._fh = open(filename)

    def __set_lengths_and_dimension(self):
        for ii, f in enumerate(self._filenames):
            self._logger.debug("set len/dim for file %s" % f)
            self.__open_file(ii)
            length = 0
            first_line = True
            # we have to read this
            for line in self._fh:
                if first_line:
                    #print line
                    arr = np.fromstring(line, sep=' ')
                    dim = arr.shape[0]
                    first_line = False
                length += dim
            self._logger.debug("calculated length: %i" %length )
            self._lengths.append(length)
            if self._ndim == 0:
                self._ndim = dim
            elif self._ndim != dim:
                raise ValueError(
                    "different dimension in current file '%s'" % f)

    def dimension(self):
        return self._ndim

    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return self._ntraj

    def trajectory_length(self, itraj, stride=1):
        """
        Returns the length of trajectory

        :param itraj:
            trajectory index
        :param stride: 
            return value is the number of frames in trajectory when
            running through it with a step size of `stride`

        :return:
            length of trajectory
        """
        return (self._lengths[itraj] - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1):
        """
        Returns the length of each trajectory

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            list containing length of each trajectory
        """
        return [(l - 1) // stride + 1 for l in self._lengths]

    def n_frames_total(self, stride=1):
        """
        Returns the total number of frames, over all trajectories

        :param stride:
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`

        :return:
            the total number of frames, over all trajectories
        """
        if stride == 1:
            return np.sum(self._lengths)
        else:
            return sum(self.trajectory_lengths(stride))

    def _reset(self, stride=1):
        self._t = 0
        self._itraj = 0
        if self._fh is not None:
            self._fh.close()

        self._current_file_is_ascii = False
        self._current_file_is_ascii = 0

    def _next_chunk(self, lag=0, stride=1):

        self.__open_file(self._itraj)

        if self._t >= self.trajectory_length(self._itraj, stride=stride):
            self._itraj += 1
            self._t = 0

        # ascii mode, read "chunksize" lines from file
        chunks = []
        count = 0
        assert not self._fh.closed
        for line in self._fh:
            chunks.append(line)
            if count >= self.chunksize + lag:
                break
            count += 1

        chunks = np.vstack(chunks)
        X = chunks[0:self.chunksize]

        if lag == 0:
            return X
        else:
            Y = chunks[lag:self.chunksize + lag]
            return X, Y
