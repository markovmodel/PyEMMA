__author__ = 'noe'

import numpy as np
from scipy.spatial.distance import cdist

from pyemma.coordinates.io.reader import ChunkedReader
from pyemma.util.log import getLogger

logger = getLogger('DataInMemory')


class DataInMemory(ChunkedReader):
    r"""
    multi-dimensional multi-trajectory data fully stored in memory

    Parameters
    ----------
    data : ndarray (nframe, ndim) or list of ndarrays (nframe, ndim) or list of filenames
        Data has to be either one 2d array which stores amount of frames in first
        dimension and coordinates/features in second dimension or a list of this
        arrays. Despite that it can also be a list of filenames (.csv or .npy files),
        which will then be lazy loaded into memory.

    """

    def __init__(self, _data, **kwargs):
        ChunkedReader.__init__(self)

        if isinstance(_data, np.ndarray):
            self.ntraj = 1
            if _data.ndim == 1:
                self.ndim = np.shape(_data)[0]
            else:
                self.ndim = np.shape(_data)[1]
            self._lengths = [np.shape(_data)[0]]
            self.data = [_data]
        elif isinstance(_data, list):
            # lazy load given filenames into memory
            if all(isinstance(d, str) for d in _data):
                self.data = []
                if 'mmap_mode' in kwargs:
                    mmap_mode = kwargs['mmap_mode']
                else:
                    mmap_mode = 'r'
                for f in _data:
                    if f.endswith('.npy'):
                        x = np.load(f, mmap_mode=mmap_mode)
                    else:
                        x = np.loadtxt(f)
                    x = np.atleast_2d(x)
                    self.data.append(x)

            # everything is an array
            elif all(isinstance(d, np.ndarray) for d in _data):
                self.data = [np.atleast_2d(d) for d in _data]
            else:
                raise ValueError("supply 2d ndarray, list of 2d ndarray"
                                 " or list of filenames storing 2d arrays.")

            self.ntraj = len(self.data)

            # ensure all trajs have same dim
            ndims = [np.shape(x)[1] for x in self.data]
            if not np.unique(ndims).size == 1:
                raise ValueError("input data has different dimensions!")

            self.ndim = ndims[0]
            self._lengths = [np.shape(d)[0] for d in self.data]
        else:
            raise ValueError('input data is neither an ndarray '
                             'nor a list of ndarrays!')

        self.t = 0
        self.itraj = 0
        self._chunksize = 0

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, x):
        # chunksize setting is forbidden, since we are operating in memory
        pass

    def number_of_trajectories(self):
        """
        Returns the number of trajectories

        :return:
            number of trajectories
        """
        return self.ntraj

    def trajectory_length(self, itraj):
        """
        Returns the length of trajectory

        :param itraj:
            trajectory index

        :return:
            length of trajectory
        """
        return self._lengths[itraj]

    def trajectory_lengths(self):
        """
        Returns the length of each trajectory

        :return:
            length of each trajectory
        """
        return self._lengths

    def n_frames_total(self):
        """
        Returns the total number of frames, over all trajectories

        :return:
            the total number of frames, over all trajectories
        """
        return np.sum(self._lengths)

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self.ndim

    def reset(self):
        """Resets the data producer
        """
        self.itraj = 0
        self.t = 0

    def next_chunk(self, lag=0):
        """

        :param lag:
        :return:
        """
        # finished once with all trajectories? so reset the pointer to allow
        # multi-pass
        if self.itraj >= self.ntraj:
            self.reset()

        traj_len = self._lengths[self.itraj]
        traj = self.data[self.itraj]

        # complete trajectory mode
        if self._chunksize == 0:
            if lag == 0:
                X = traj
                self.itraj += 1
                return X
            else:
                X = traj
                Y = traj[lag:traj_len]
                self.itraj += 1
                return (X, Y)

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
