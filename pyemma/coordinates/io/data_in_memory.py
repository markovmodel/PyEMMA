__author__ = 'noe'

import numpy as np
from pyemma.coordinates.io.reader import ChunkedReader
from pyemma.util.log import getLogger

logger = getLogger('DataInMemory')


class DataInMemory(ChunkedReader):

    """
    multi-dimensional multi-trajectory data fully stored in memory
    """

    def __init__(self, _data):
        """

        :param data:
            ndarray of shape (nframe, ndim) or
            list of ndarrays, each of shape (nframe_i, ndim)
        """
        ChunkedReader.__init__(self)

        if isinstance(_data, np.ndarray):
            self.data = [_data]
            self.ntraj = 1
            if _data.ndim == 1:
                self.ndim = np.shape(_data)[0]
            else:
                self.ndim = np.shape(_data)[1]
            self._lengths = [np.shape(_data)[0]]
        elif isinstance(_data, list):
            self.data = _data
            self.ntraj = len(_data)
            # ensure all trajs have same dim
            ndims = np.fromiter(([np.shape(_data[i])[1]
                                  for i in xrange(len(_data))]), dtype=int)
            ndim_eq = ndims == np.shape(_data[0][1])
            if not np.all(ndim_eq):
                raise ValueError("input data has different dimensions!"
                                 " Indices not matching: %s"
                                 % np.where(ndim_eq == False))
            self.ndim = ndims[0]

            self._lengths = [np.shape(_data[i])[0] for i in range(len(_data))]
        else:
            raise ValueError('input data is neither an ndarray '
                             'nor a list of ndarrays!')

        self.t = 0
        self.itraj = 0
        self.chunksize = 0

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
        if self.chunksize == 0:
            if lag == 0:
                X = traj
                self.itraj += 1
                return X
            else:
                X = traj[: -lag]
                Y = traj[lag:traj_len]
                self.itraj += 1
                return (X, Y)
        else:
            #logger.debug("t=%i" % self.t)
            # FIXME: if t + chunksize < traj_len, this selects wrong bounds. eg [100:40], which is empty
            chunksize_bounds = min(self.t + self.chunksize, traj_len)
            if lag == 0:
                X = traj[self.t:chunksize_bounds]
                self.t += np.shape(X)[0]
                if self.t >= traj_len:
                    self.itraj += 1
                return X
            else:
                logger.warning("chunked lagged access not debugged!")
                X = traj[self.t: chunksize_bounds - lag]
                assert np.shape(X)[0] > 0
                #logger.debug("Y=traj[%i+%i : %i]" %
                #             (self.t, lag, chunksize_bounds))
                # if we do not have enough data anymore for chunked, padd it with zeros
                Y = traj[self.t + lag: chunksize_bounds]
#                 if np.shape(Y)[0] == 0:
#                     assert False
#                     Y = PaddedArray(np.zeros_like(X), X.shape)

                assert np.shape(X) == np.shape(Y), "%s != %s" % (str(np.shape(X)), str(np.shape(Y)))
                self.t += np.shape(X)[0]
                assert np.shape(Y)[0] > 0
                if self.t + lag >= traj_len:
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
        return np.linalg.norm(Y - x, 2, axis=1)