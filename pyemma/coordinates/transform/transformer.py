__author__ = 'noe'

from pyemma.util.log import getLogger

import numpy as np


log = getLogger('Transformer')
__all__ = ['Transformer']


class Transformer(object):

    """

    Parameters
    ----------
    chunksize : int (optional)
        the chunksize used to batch process underlying data
    lag : int (optional)
        if you want to process time lagged data, set this to a value > 0.
    """

    def __init__(self, chunksize=100, lag=0):
        self.chunksize = chunksize
        self._lag = lag
        self._in_memory = False
        self._dataproducer = None

    @property
    def data_producer(self):
        """where does this transformer gets its data from"""
        return self._dataproducer

    @data_producer.setter
    def data_producer(self, dp):
        self._dataproducer = dp

    @property
    def chunksize(self):
        """chunksize defines how much data is being processed at once."""
        return self._chunksize

    @chunksize.setter
    def chunksize(self, size):
        assert size >= 0, "chunksize has to be positive"
        self._chunksize = int(size)

    @property
    def in_memory(self):
        """are results stored in memory?"""
        return self._in_memory

    def operate_in_memory(self):
        """
        If called, the output will be stored in memory
        """
        self._in_memory = True
        # output data
        self.Y = [np.zeros((self.trajectory_length(itraj), self.dimension()))
                  for itraj in xrange(self.number_of_trajectories())]

    @property
    def lag(self):
        """lag time, at which a second time lagged data source will be processed.
        """
        return self._lag

    @lag.setter
    def lag(self, lag):
        assert lag >= 0, "lag time has to be positive."
        self._lag = int(lag)

    def number_of_trajectories(self):
        """
        Returns the number of trajectories.

        Returns
        -------
            int : number of trajectories
        """
        return self.data_producer.number_of_trajectories()

    def trajectory_length(self, itraj):
        """
        Returns the length of trajectory with given index.

        Parameters
        ----------
        itraj : int
            trajectory index

        Returns
        -------
        int : length of trajectory
        """
        return self.data_producer.trajectory_length(itraj)

    def trajectory_lengths(self):
        """
        Returns the length of each trajectory.

        Returns
        -------
        int : length of each trajectory
        """
        return self.data_producer.trajectory_lengths()

    def n_frames_total(self):
        """
        Returns total number of frames.

        Returns
        -------
        int : n_frames_total
        """
        return self.data_producer.n_frames_total()

    def get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes
        """
        return 4 * self.dimension()

    def get_constant_memory(self):
        """Returns the constant memory requirements, in bytes."""
        return 0

    def describe(self):
        """ get a representation of this Transformer"""
        return self.__str__()

    def parametrize(self):
        r""" parametrize this Transformer
        """
        # check if ready
        if self.data_producer is None:
            raise RuntimeError('Called parametrize of %s while data producer is not'
                               ' yet set. Ensure "data_producer" attribute is set!'
                               % self.describe())
        # init
        self.param_init()
        # feed data, until finished
        add_data_finished = False
        ipass = 0

        # parametrize
        while not add_data_finished:
            first_chunk = True
            self.data_producer.reset()
            # iterate over trajectories
            last_chunk = False
            itraj = 0
            lag = self._lag
            while not last_chunk:
                last_chunk_in_traj = False
                t = 0
                while not last_chunk_in_traj:
                    # iterate over times within trajectory
                    if lag == 0:
                        X = self.data_producer.next_chunk()
                        Y = None
                    else:
                        if self.trajectory_length(itraj) <= lag:
                            log.error(
                                "trajectory nr %i to short, skipping it" % self.itraj)
                            break
                        X, Y = self.data_producer.next_chunk(lag=lag)
                    L = np.shape(X)[0]
                    # last chunk in traj?
                    last_chunk_in_traj = (
                        t + lag + L >= self.trajectory_length(itraj))
                    # last chunk?
                    last_chunk = (
                        last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                    # first chunk
                    add_data_finished = self.param_add_data(
                        X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=Y)
                    first_chunk = False
                    # increment time
                    t += L
                # increment trajectory
                itraj += 1
            ipass += 1
        # finish parametrization
        self.param_finish()
        # memory mode? Then map all results
        if self.in_memory:
            self.map_to_memory()

    def param_init(self):
        """
        Initializes the parametrization.
        """
        # create mean array and covariance matrix
        pass

    def param_finish(self):
        """
        Finalizes the parametrization.
        """
        pass

    def map_to_memory(self):
        """maps results to memory. Will be stored in attribute :attr:`Y`."""
        # if operating in main memory, do all the mapping now
        self.data_producer.reset()
        # iterate over trajectories
        last_chunk = False
        itraj = 0
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                X = self.data_producer.next_chunk()
                L = np.shape(X)[0]
                # last chunk in traj?
                last_chunk_in_traj = (t + L >= self.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                # write
                self.Y[itraj][t:t + L] = self.map(X)
                # increment time
                t += L
            # increment trajectory
            itraj += 1

    def reset(self):
        """reset data position"""
        if self.in_memory:
            # operate in memory, implement iterator here
            self.itraj = 0
            self.t = 0
        else:
            # operate in pipeline
            self.data_producer.reset()

    def next_chunk(self, lag=0):
        """
        transforms next available chunk from either in memory data or internal
        data_producer

        Parameters
        ----------
        lag  : int
            time delay of second data source.

        Returns
        -------
        X, (Y if lag > 0) : array_like
            mapped (transformed) data
        """
        if self.in_memory:
            if self.itraj >= self.number_of_trajectories():
                return None
            # operate in memory, implement iterator here
            traj_len = self.trajectory_length(self.itraj)
            if lag == 0:
                Y = self.Y[self.itraj][
                    self.t:min(self.t + self.chunksize, traj_len)]
                # increment counters
                self.t += self.chunksize
                if self.t >= traj_len:
                    self.itraj += 1
                    self.t = 0
                return Y
            else:
                Y0 = self.Y[self.itraj][
                    self.t:min(self.t + self.chunksize, traj_len)]
                Ytau = self.Y[self.itraj][
                    self.t + lag:min(self.t + self.chunksize + lag, traj_len)]
                # increment counters
                self.t += self.chunksize
                if self.t >= traj_len:
                    self.itraj += 1
                return (Y0, Ytau)
        else:
            # operate in pipeline
            if lag == 0:
                X = self.data_producer.next_chunk()
                return self.map(X)
            else:
                (X0, Xtau) = self.data_producer.next_chunk(lag=lag)
                return (self.map(X0), self.map(Xtau))

    def __iter__(self):
        self.reset()
        self.last_chunk = False
        self.itraj = 0
        self.t = 0
        return self

    def next(self):
        """ enable iteration over transformed data.

        Returns
        -------
        (itraj, X) : (int, ndarray(n, m)
            itraj corresponds to input sequence number (eg. trajectory index)
            and X is the transformed data, n = chunksize or n < chunksize at end
            of input.

        """
        # iterate over trajectories
        if self.itraj >= self.number_of_trajectories():
            raise StopIteration

        X = self.data_producer.next_chunk()
        L = np.shape(X)[0]
        self.t += L
        last_itraj = self.itraj
        if self.t >= self.trajectory_length(self.itraj):
            self.itraj += 1
            self.t = 0
        return (last_itraj, self.map(X))

    @staticmethod
    def distance(x, y):
        """ stub for calculating the euclidian norm between x and y.

        Parameters
        ----------
        x : ndarray(n)
        y : ndarray(n)

        Returns
        -------
        d : float
            euclidean distance
        """
        return np.linalg.norm(x - y, 2)

    @staticmethod
    def distances(x, Y):
        """ stub for calculating the euclidian norm between x and a set of points Y.

        Parameters
        ----------
        x : ndarray(n)
        Y : ndarray(m, n)

        Returns
        -------
        distances : ndarray(m)
            euclidean distances between points in Y to x
        """
        return np.linalg.norm(Y - x, 2, axis=1)
