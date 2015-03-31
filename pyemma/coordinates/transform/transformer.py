__author__ = 'noe'

from pyemma.util.log import getLogger

import numpy as np
from scipy.spatial.distance import cdist

log = getLogger('Transformer')
__all__ = ['Transformer']


class Transformer(object):

    """ Basis class for pipeline objects

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
        self._parametrized = False

    @property
    def data_producer(self):
        """where does this transformer gets its data from"""
        return self._dataproducer

    @data_producer.setter
    def data_producer(self, dp):
        if dp is not self._dataproducer:
            self._parametrized = False
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

    @in_memory.setter
    def in_memory(self, op_in_mem):
        """
        If called, the output will be stored in memory
        """
        if not self._in_memory and op_in_mem:
            self.Y = [np.zeros((self.trajectory_length(itraj), self.dimension()))
                      for itraj in xrange(self.number_of_trajectories())]
        elif not op_in_mem and self._in_memory:
            self._clear_in_memory()

        self._in_memory = op_in_mem

    def _clear_in_memory(self):
        assert self.in_memory, "tried to delete in memory results which are not set"
        for y in self.Y:
            del y

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

    def _get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes
        """
        return 4 * self.dimension()

    def _get_constant_memory(self):
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

        if self._parametrized:
            return

        # init
        self._param_init()
        # feed data, until finished
        add_data_finished = False
        ipass = 0

        # parametrize
        while not add_data_finished:
            first_chunk = True
            self.data_producer._reset()
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
                        X = self.data_producer._next_chunk()
                        Y = None
                    else:
                        if self.trajectory_length(itraj) <= lag:
                            log.error(
                                "trajectory nr %i to short, skipping it" % itraj)
                            break
                        X, Y = self.data_producer._next_chunk(lag=lag)
                    L = np.shape(X)[0]
                    # last chunk in traj?
                    last_chunk_in_traj = (
                        t + L >= self.trajectory_length(itraj))
                    # last chunk?
                    last_chunk = (
                        last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                    # first chunk
                    add_data_finished = self._param_add_data(
                        X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=Y)
                    first_chunk = False
                    # increment time
                    t += L
                # increment trajectory
                itraj += 1
            ipass += 1
        # finish parametrization
        self._param_finish()
        self._parametrized = True
        # memory mode? Then map all results
        if self.in_memory:
            self._map_to_memory()

    def map(self, X):
        """Maps the input data through the transformer to correspondingly shaped output data.

        Parameters
        ----------
        X : ndarray(T, n) or list of ndarray(T_i, n)
            The input data, where T is the number of time steps and n is the number of dimensions.
            When a list is provided they can have differently many time steps, but the number of dimensions need
            to be consistent.

        Returns
        -------
        Y : ndarray(T, d) or list of ndarray(T_i, d)
            the mapped data, where T is the number of time steps of the input data and d is the output dimension
            of this transformer. If called with a list of trajectories, Y will also be a corresponding list of
            trajectories
        """
        # TODO: This is a very naive implementation of case switching. Please check and make more robust if needed.
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                return self._map_array(X)
            else:
                raise TypeError('Input has the wrong shape: '+str(X.shape)+' with '+str(X.ndim)+' dimensions. Expecting a matrix (2 dimensions)')
        elif isinstance(X, list):
            out = []
            for x in X:
                out.append(self._map_array(x))
        else:
            raise TypeError('Input has the wrong type: '+str(type(X))+'. Either accepting numpy arrays of dimension 2 or lists of such arrays')

    # TODO: implement
    def get_output(self, stride=1):
        """Maps all input data of this transformer and returns it as and array or list of arrays

        Parameters
        ----------
        stride : int, optional, default = 1
            If set to 1, all frames of the input data will be read and mapped. This gives you great detail, but might
            be slow and create memory issues when trying to allocate the resulting output array.
            If set greater than 1, only every so many frames will be read and mapped. The output arrays are
            correspondingly smaller

        Returns:
        --------
        output : ndarray(T, d) or list of ndarray(T_i, d)
            the mapped data, where T is the number of time steps of the input data, or if stride > 1,
            floor(T_in / stride). d is the output dimension of this transformer.
            If the input consists of a list of trajectories, Y will also be a corresponding list of trajectories

        """
        pass

    def _map_array(self, X):
        """
        Initializes the parametrization.

        Parameters
        ----------
        X : ndarray(T, n)
            The input data, where T is the number of time steps and n is the number of dimensions.

        Returns
        -------
        Y : ndarray(T, d)
            the projected data, where T is the number of time steps of the input data and d is the output dimension
            of this transformer.

        """
        pass


    def _param_init(self):
        """
        Initializes the parametrization.
        """
        pass

    def _param_finish(self):
        """
        Finalizes the parametrization.
        """
        pass

    def _map_to_memory(self):
        """maps results to memory. Will be stored in attribute :attr:`Y`."""
        # if operating in main memory, do all the mapping now
        self.data_producer._reset()
        # iterate over trajectories
        last_chunk = False
        itraj = 0
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                X = self.data_producer._next_chunk()
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

    def _reset(self):
        """_reset data position"""
        if not self._parametrized:
            self.parametrize()
        self._itraj = 0
        self._t = 0
        if not self.in_memory:
            # operate in pipeline
            self.data_producer._reset()

    def _next_chunk(self, lag=0):
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
            if self._itraj >= self.number_of_trajectories():
                return None
            # operate in memory, implement iterator here
            traj_len = self.trajectory_length(self._itraj)
            if lag == 0:
                Y = self.Y[self._itraj][
                    self._t:min(self._t + self.chunksize, traj_len)]
                # increment counters
                self._t += self.chunksize
                if self._t >= traj_len:
                    self._itraj += 1
                    self._t = 0
                return Y
            else:
                Y0 = self.Y[self._itraj][
                    self._t:min(self._t + self.chunksize, traj_len)]
                Ytau = self.Y[self._itraj][
                    self._t + lag:min(self._t + self.chunksize + lag, traj_len)]
                # increment counters
                self._t += self.chunksize
                if self._t >= traj_len:
                    self._itraj += 1
                return (Y0, Ytau)
        else:
            # operate in pipeline
            if lag == 0:
                X = self.data_producer._next_chunk()
                self._t += X.shape[0]
                return self.map(X)
            else:
                (X0, Xtau) = self.data_producer._next_chunk(lag=lag)
                self._t += X0.shape[0]
                return (self.map(X0), self.map(Xtau))

    def __iter__(self):
        self._reset()
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
        if self._itraj >= self.number_of_trajectories():
            raise StopIteration

        # next chunk already maps output
        if self.lag == 0:
            X = self._next_chunk()
        else:
            X, Y = self._next_chunk(self.lag)

        last_itraj = self._itraj
        # note: _t is incremented in _next_chunk
        if self._t >= self.trajectory_length(self._itraj):
            self._itraj += 1
            self._t = 0

        return (last_itraj, X)

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
        x = np.atleast_2d(x)
        dists = cdist(Y, x)
        return dists
