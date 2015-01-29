__author__ = 'noe'

import numpy as np

class DataInMemory:
    """
    multi-dimensional multi-trajectory data fully stored in memory
    """
    # FIXME: if defined like this, those attributes are static! they should go to ctor
    # data
    data = []
    ntraj = 0
    ndim = 0
    lengths = []

    # chunking, lagging
    chunksize = 0
    lag = 0

    # count
    itraj = 0
    t = 0

    def __init__(self, _data):
        """

        :param data:
            ndarray of shape (nframe, ndim) or
            list of ndarrays, each of shape (nframe_i, ndim)
        :return:
        """
        if isinstance(_data, np.ndarray):
            self.data = [_data]
            self.ntraj = 1
            self.ndim = np.shape(_data)[1]
            self.lengths = [np.shape(_data)[0]]
        elif isinstance(_data, (list)):
            self.data = [_data]
            self.ntraj = len(_data)
            self.ndim = np.shape(_data[0])[1]
            self.lengths = [np.shape(_data[i])[0] for i in range(len(_data))]
        else:
            raise ValueError('input data is neither an ndarray nor a list of ndarrays')


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
        return self.lengths[itraj]


    def trajectory_lengths(self):
        """
        Returns the length of each trajectory

        :return:
            length of each trajectory
        """
        return self.lengths



    def n_frames_total(self):
        """
        Returns the total number of frames, over all trajectories

        :return:
            the total number of frames, over all trajectories
        """
        return np.sum(self.lengths)


    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self.ndim


    def set_chunksize(self, size):
        """
        Sets the size of data chunks that are read to memory at one time.

        :param size:
        :return:
        """
        self.chunksize = size


    def reset(self):
        """
        Resets the data producer

        :return:
        """
        self.itraj = 0
        self.t = 0


    def next_chunk(self):
        """

        :param lag:
        :return:
        """
        # finished?
        if self.itraj >= self.ntraj:
            return None
        # complete trajectory mode
        if self.chunksize == 0:
            if self.lag == 0:
                X = self.data[self.itraj]
                self.itraj += 1
                return X
            else:
                X = self.data[self.itraj][0:self.lengths[self.itraj]-self.lag]
                Y = self.data[self.itraj][self.lag:self.lengths[self.itraj]]
                self.itraj += 1
                return (X,Y)
        else:
            if self.lag == 0:
                X = self.data[self.itraj][self.t:min(self.t+self.chunksize,self.lengths[self.itraj])]
                self.t += self.chunksize
                if self.t >= self.lengths(self.itraj):
                    self.itraj += 1
                return X
            else:
                X = self.data[self.itraj][self.t:min(self.t+self.chunksize,self.lengths[self.itraj])-self.lag]
                Y = self.data[self.itraj][self.t+self.lag:min(self.t+self.chunksize,self.lengths[self.itraj])]
                self.t += self.chunksize
                if self.t + self.lag >= self.lengths(self.itraj):
                    self.itraj += 1
                return (X,Y)



