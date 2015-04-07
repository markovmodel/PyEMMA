from pyemma.coordinates.io.file_reader import FileReader
__author__ = 'noe, marscher'

import numpy as np
import functools

from scipy.spatial.distance import cdist

from pyemma.coordinates.transform.transformer import Transformer

# TODO: consider having this with an underlying loader, since we might now want
# everyting in memory!


class DataInMemory(Transformer):

    r"""
    multi-dimensional data fully stored in memory.

    Used to pass arbitrary coordinates to pipeline. Data is being flattened to 
    two dimensions to ensure it is compatible.

    Parameters
    ----------
    data : ndarray (nframe, ndim) or list of ndarrays (nframe, ndim)
        Data has to be either one 2d array which stores amount of frames in first
        dimension and coordinates/features in second dimension or a list of this
        arrays.
    """

    def __init__(self, data):
        Transformer.__init__(self, chunksize=1000)
        self.data_producer = self

        # storage
        self._data = []

        if not isinstance(data, (list, tuple)):
            data = [data]

        # everything is an array
        if all(isinstance(d, np.ndarray) for d in data):
            for d in data:
                self.__add_array_to_storage(d)
        else:
            raise ValueError("supply 2d ndarray, list of 2d ndarray"
                             " or list of filenames storing 2d arrays.")

        self.__set_dimensions_and_lenghts()

        # internal counters
        self._t = 0
        self._itraj = 0

        self._parametrized = True

    @classmethod
    def load_from_files(cls, files):
        """ construct this by loading all files into memory

        Parameters
        ----------
        files: str or list of str
            filenames to read from
        """
        reader = FileReader(files)
        data = reader.get_output()
        return cls(data)

    def __add_array_to_storage(self, array):
        # checks shapes, eg convert them (2d), raise if not possible
        # after checks passed, add array to self._data

        if array.ndim == 1:
            array = np.atleast_2d(array).T
        elif array.ndim == 2:
            pass
        else:
            shape = array.shape
            # hold first dimension, multiply the rest
            shape_2d = (
                shape[0], functools.reduce(lambda x, y: x * y, shape[1:]))
            array = np.reshape(array, shape_2d)

        self._data.append(array)

    def __set_dimensions_and_lenghts(self):
        # number of trajectories/data sets
        self._ntraj = len(self._data)
        if self._ntraj == 0:
            raise ValueError("no valid data")

        # this works since everything is flattened to 2d
        self._lengths = [np.shape(d)[0] for d in self._data]

        # ensure all trajs have same dim
        ndims = [np.shape(x)[1] for x in self._data]
        if not np.unique(ndims).size == 1:
            raise ValueError("input data has different dimensions!"
                             "Dimensions are = %s" % ndims)

        self._ndim = ndims[0]

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
        # FIXME: in case of 1 traj, this returns 1!!!
        if stride == 1:
            self._logger.debug("self._lengths= %s " % self._lengths)
            return np.sum(self._lengths)
        else:
            return sum(self.trajectory_lengths(stride))

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self._ndim

    def _reset(self, stride=1):
        """Resets the data producer
        """
        self._itraj = 0
        self._t = 0

    def _next_chunk(self, lag=0, stride=1):
        """

        :param lag:
        :return:
        """
        # finished once with all trajectories? so _reset the pointer to allow
        # multi-pass
        if self._itraj >= self._ntraj:
            self._reset()

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
            upper_bound = min(self._t + (self._chunksize + 1)*stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)

            X = traj[slice_x]
            self._logger.debug(X[0])

            if lag == 0:
                self._t = upper_bound

                if upper_bound >= traj_len:
                    self._itraj += 1
                    self._t = 0
                return X
            else:
                # its okay to return empty chunks
                upper_bound = min(self._t + (lag + self._chunksize+1)*stride, traj_len)
                slice_y = slice(self._t + lag, upper_bound, stride)
                self._t += X.shape[0]

                if self._t >= traj_len:
                    self._itraj += 1
                    self._t = 0
                Y = traj[slice_y]
                return X, Y

    @staticmethod
    def distance(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return np.linalg.norm(x - y, 2)

    @staticmethod
    def distances(x, Y):
        """

        :param x: ndarray (n)
        :param y: ndarray (Nxn)
        :return:
        """
        dists = cdist(Y, x)
        return dists
