__author__ = 'noe, marscher'

import numpy as np
import functools

from scipy.spatial.distance import cdist

from pyemma.coordinates.transform.transformer import Transformer
from pyemma.util.log import getLogger

# TODO: consider having this with an underlying loader, since we might now want
# everyting in memory!


class DataInMemory(Transformer):

    r"""
    multi-dimensional data fully stored in memory.

    Used to pass arbitrary coordinates to pipeline. Data is being flattened to 
    two dimensions to ensure it is compatible.

    Parameters
    ----------
    data : ndarray (nframe, ndim) or list of ndarrays (nframe, ndim) or list of filenames
        Data has to be either one 2d array which stores amount of frames in first
        dimension and coordinates/features in second dimension or a list of this
        arrays. Despite that it can also be a list of filenames (.csv or .np[y,z] files),
        which will then be loaded into memory.

    mmap_mode : str (optional), default='r'
        binary NumPy arrays are being memory mapped using this flag.

    """

    def __init__(self, data, mmap_mode='r'):
        Transformer.__init__(self, chunksize=1000)
        self.logger = getLogger('DataInMemory[%i]' % id(self))
        self.data_producer = self

        # storage
        self._data = []

        self.mmap_mode = mmap_mode

        if not isinstance(data, (list, tuple)):
            data = [data]

        # files (path strings)
        if all(isinstance(d, basestring) for d in data):

            # check files are readable via pre-creating file handles
            fh = [open(f) for f in data]

            for ii, f in enumerate(data):
                if f.endswith('.npy'):
                    x = np.load(f, mmap_mode=mmap_mode)
                    fh[ii].close()
                    self.__add_array_to_storage(x)
                elif f.endswith('.npz'):
                    # closes file handle
                    with np.load(fh[ii]) as fh:
                        for _, x in fh.items():
                            self.__add_array_to_storage(x)
                else:
                    x = np.loadtxt(fh[ii])
                    fh[ii].close()
                    self.__add_array_to_storage(x)

        # everything is an array
        elif all(isinstance(d, np.ndarray) for d in data):
            for d in data:
                self.__add_array_to_storage(d)
        else:
            raise ValueError("supply 2d ndarray, list of 2d ndarray"
                             " or list of filenames storing 2d arrays.")

        self.__set_dimensions_and_lenghts()

        # internal counters
        self._t = 0
        self._itraj = 0

    def __add_array_to_storage(self, array):
        # checks shapes, eg convert them (2d), raise if not possible
        # after checks passed, add array to self._data

        if array.ndim == 1:
            array = np.atleast_2d(array)
        elif array.ndim == 2:
            pass
        else:
            shape = array.shape
            # hold first dimension, multiply the rest
            shape_2d = (shape[0], functools.reduce(lambda x, y: x * y, shape[1:]))
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

    def parametrize(self):
        self._parametrized = True

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
            upper_bound = min(self._t + self._chunksize * stride, traj_len)
            slice_x = slice(self._t, upper_bound, stride)

            X = traj[slice_x]
            self._t += X.shape[0]

            if self._t >= traj_len:
                self._itraj += 1
                self._t = 0

            if lag == 0:
                return X
            else:
                # its okay to return empty chunks
                upper_bound = min(
                    self._t + (lag + self._chunksize) * stride, traj_len)
                Y = traj[self._t + lag: upper_bound: stride]
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
