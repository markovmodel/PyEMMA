__author__ = 'noe'

from pyemma.util.log import getLogger

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ['Transformer']


class TransformerIterator(object):
    def __init__(self, transformer, stride=1, lag=0):
        # reset transformer iteration

        # TODO: I think it might be necessary to reset the transformer before starting to iterate. If not, can you
        # TODO: guarantee it is set to the correct initial settings such as _itraj=0?
        # transformer._reset()
        self._stride = stride
        self._lag = lag
        self._transformer = transformer

    def __iter__(self):
        return self

    def next(self):
        if self._transformer._itraj >= self._transformer.number_of_trajectories():
            raise StopIteration

        last_itraj = self._transformer._itraj
        if self._lag == 0:
            X = self._transformer._next_chunk(lag=self._lag, stride=self._stride)
            return (last_itraj, X)
        else:
            X, Y = self._transformer._next_chunk(lag=self._lag, stride=self._stride)
            return (last_itraj, X, Y)


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
        self._in_memory = False
        self._dataproducer = None
        self._parametrized = False

        self.__create_logger()

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
            self._map_to_memory()
        elif not op_in_mem and self._in_memory:
            self._clear_in_memory()

        self._in_memory = op_in_mem

    def _clear_in_memory(self):
        assert self.in_memory, "tried to delete in memory results which are not set"
        for y in self.Y:
            del y

    def dimension(self):
        """ output dimension of this transformation """
        raise NotImplementedError('this method has to be implemented in'
                                  ' children of this class.')

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def number_of_trajectories(self):
        """
        Returns the number of trajectories.

        Returns
        -------
            int : number of trajectories
        """
        return self.data_producer.number_of_trajectories()

    def trajectory_length(self, itraj, stride=1):
        """
        Returns the length of trajectory with given index.

        Parameters
        ----------
        itraj : int
            trajectory index
        stride : int
            return value is the number of frames in trajectory when
            running through it with a step size of `stride`            

        Returns
        -------
        int : length of trajectory
        """
        return self.data_producer.trajectory_length(itraj, stride=stride)

    def trajectory_lengths(self, stride=1):
        """
        Returns the length of each trajectory.

        Parameters
        ----------
        stride : int
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`        
        Returns
        -------
        int : length of each trajectory
        """
        return self.data_producer.trajectory_lengths(stride=stride)

    def n_frames_total(self, stride=1):
        """
        Returns total number of frames.

        Parameters
        ----------
        stride : int
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`        
        Returns
        -------
        int : n_frames_total
        """
        return self.data_producer.n_frames_total(stride=stride)

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

    def dimension(self):
        return 0  # default

    def output_type(self):
        return np.float32

    def parametrize(self, stride=1):
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
        return_value = self._param_init()
        if return_value is not None:
            lag = return_value
        else:
            lag = 0
        # feed data, until finished
        add_data_finished = False
        ipass = 0

        # parametrize
        while not add_data_finished:
            first_chunk = True
            self.data_producer._reset(stride=stride)
            # iterate over trajectories
            last_chunk = False
            itraj = 0
            # lag = self._lag
            while not last_chunk:
                last_chunk_in_traj = False
                t = 0
                while not last_chunk_in_traj:
                    # iterate over times within trajectory
                    if lag == 0:
                        X = self.data_producer._next_chunk(stride=stride)
                        Y = None
                    else:
                        X, Y = self.data_producer._next_chunk(lag=lag, stride=stride)
                    L = np.shape(X)[0]
                    # last chunk in traj?
                    last_chunk_in_traj = (
                        t + L >= self.trajectory_length(itraj, stride=stride))
                    # last chunk?
                    last_chunk = (
                        last_chunk_in_traj and itraj >= self.number_of_trajectories() - 1)
                    # first chunk
                    return_value = self._param_add_data(
                        X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=Y, stride=stride)
                    if isinstance(return_value, tuple):
                        add_data_finished, lag = return_value
                    else:
                        add_data_finished = return_value
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
                mapped = self._map_array(X)
                return mapped
            else:
                raise TypeError('Input has the wrong shape: '+str(X.shape)+' with '+str(X.ndim)
                                +' dimensions. Expecting a matrix (2 dimensions)')
        elif isinstance(X, list):
            out = []
            for x in X:
                mapped = self._map_array(x)
                out.append(mapped)
        else:
            raise TypeError('Input has the wrong type: '+str(type(X))
                            +'. Either accepting numpy arrays of dimension 2 or lists of such arrays')

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

    def _reset(self, stride=1):
        """_reset data position"""
        if not self._parametrized:  # TODO: should this stay or should it go?
            self.parametrize()
        self._itraj = 0
        self._t = 0
        if not self.in_memory and self.data_producer is not self:
            # operate in pipeline
            self.data_producer._reset(stride=stride)

    def _next_chunk(self, lag=0, stride=1):
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
                    self._t:min(self._t + self.chunksize*stride, traj_len):stride]
                # increment counters
                self._t += self.chunksize*stride
                if self._t >= traj_len:
                    self._itraj += 1
                    self._t = 0
                return Y
            else:
                Y0 = self.Y[self._itraj][
                    self._t:min(self._t + self.chunksize*stride, traj_len):stride]
                Ytau = self.Y[self._itraj][
                    self._t + lag*stride:min(self._t + (self.chunksize + lag)*stride, traj_len):stride]
                # increment counters
                self._t += self.chunksize*stride
                if self._t >= traj_len:
                    self._itraj += 1
                return (Y0, Ytau)
        else:
            # operate in pipeline
            if lag == 0:
                X = self.data_producer._next_chunk(stride=stride)
                self._t += X.shape[0]
                if self._t >= self.trajectory_length(self._itraj, stride=stride):
                    self._itraj += 1
                    self._t = 0
                return self.map(X)
            else:
                (X0, Xtau) = self.data_producer._next_chunk(lag=lag, stride=stride)
                self._t += X0.shape[0]
                if self._t >= self.trajectory_length(self._itraj, stride=stride):
                    self._itraj += 1
                    self._t = 0                
                return (self.map(X0), self.map(Xtau))

    def __iter__(self):
        """
        Returns an iterator that allows to access the transformed data.
        
        Returns
        -------
        iterator : `pyemma.coordinates.transfrom.TransformerIterator`
            a call to the .next() method of this iterator will return the pair
            (itraj, X) : (int, ndarray(n, m))
            where itraj corresponds to input sequence number (eg. trajectory index)
            and X is the transformed data, n = chunksize or n < chunksize at end
            of input.
        """        
        self._reset()
        return TransformerIterator(self, stride=1, lag=0)

    def iterator(self, stride=1, lag=0):
        """
        Returns an iterator that allows to access the transformed data.
        
        Parameters
        ----------
        stride : int
            Only transform every N'th frame, default = 1
        lag : int
            Configure the iterator such that it will return time-lagged data
            with a lag time of `lag`. If `lag` is used together with `stride`
            the operation will work as if the striding operation is applied
            before the time-lagged trajectory is shifted by `lag` steps.
            Therefore the effective lag time will be stride*lag.

        Returns
        -------
        iterator : `pyemma.coordinates.transfrom.TransformerIterator`
            If lag = 0, a call to the .next() method of this iterator will return
            the pair
            (itraj, X) : (int, ndarray(n, m))
            where itraj corresponds to input sequence number (eg. trajectory index)
            and X is the transformed data, n = chunksize or n < chunksize at end
            of input.

            If lag > 0, a call to the .next() method of this iterator will return
            the tuple
            (itraj, X, Y) : (int, ndarray(n, m), ndarray(p, m))
            where itraj and X are the same as above and Y contain the time-lagged
            data.
        """
        self._reset(stride=stride)
        return TransformerIterator(self, stride=stride, lag=lag)

    def get_output(self, dimensions=slice(0, None), stride=1):
        """ Maps all input data of this transformer and returns it as an array or list of arrays

            Parameters
            ----------
            transfrom : pyemma.coordinates.transfrom.Transformer object
                transform that provides the input data
            dimensions : list-like of indexes or slice
                indices of dimensions you like to keep, default = all
            stride : int
                only take every n'th frame, default = 1

            Returns
            -------
            output : ndarray(T, d) or list of ndarray(T_i, d)
                the mapped data, where T is the number of time steps of the input data, or if stride > 1,
                floor(T_in / stride). d is the output dimension of this transformer.
                If the input consists of a list of trajectories, Y will also be a corresponding list of trajectories

            Notes
            -----
            This function may be RAM intensive if stride is too large or
            too many dimensions are selected.

            Example
            -------
            plotting trajectories

            >>> import pyemma.coordinates as coor
            >>> import matplotlib.pyplot as plt
            >>> %matplotlib inline # only for ipython notebook
            >>>
            >>> tica = coor.tica() # fill with some actual data!
            >>> trajs = tica.get_output(dimensions=(0,), stride=100)
            >>> for traj in trajs:
            >>>     plt.figure()
            >>>     plt.plot(traj[:, 0])

        """

        if isinstance(dimensions, int):
            ndim = 1
            dimensions = slice(dimensions, dimensions+1)
        elif isinstance(dimensions, list):
            ndim = len(np.zeros(self.dimension())[dimensions])
        elif isinstance(dimensions, np.ndarray):
            assert dimensions.ndim == 1, 'dimension indices can\'t have more than one dimension'
            ndim = len(np.zeros(self.dimension())[dimensions])
        elif isinstance(dimensions, slice):
            ndim = len(np.zeros(self.dimension())[dimensions])
        else:
            raise Exception('unsupported type (%s) of \"dimensions\"' % type(dimensions))

        assert ndim > 0, "ndim was zero in %s" % self.__class__.__name__

        # allocate memory
        trajs = [np.empty((l, ndim), dtype=self.output_type()) for l in self.trajectory_lengths(stride=stride)]

        if __debug__:
            self._logger.debug("get_output(): created output trajs with shapes: %s"
                               % [x.shape for x in trajs])

        # fetch data
        last_itraj = -1
        t = 0  # first time point
        for itraj, chunk in self.iterator(stride=stride):
            if itraj != last_itraj:
                last_itraj = itraj
                t = 0  # reset time to 0 for new trajectory
            L = chunk.shape[0]
            trajs[itraj][t:t + L, :] = chunk[:, dimensions]
            t += L

        return trajs

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
