__author__ = 'noe'

from pyemma.coordinates.clustering.interface import AbstractClustering

import numpy as np

__all__ = ['UniformTimeClustering']


class UniformTimeClustering(AbstractClustering):

    """
    Uniform time clustering

    Parameters
    ----------
    k : int
    """

    def __init__(self, k=2):
        super(UniformTimeClustering, self).__init__()
        self.k = k

    def describe(self):
        return "[Uniform time clustering, k = %i]" % self.k

    def _get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes

        :return:
        """
        # 4 bytes per frame for an integer index
        return 0

    def _get_constant_memory(self):
        """
        Returns the constant memory requirements, in bytes

        :return:
        """
        # memory for cluster centers and discrete trajectories
        return self.k * 4 * self.data_producer.dimension() + 4 * self.data_producer.n_frames_total()

    def _param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        self._logger.info("Running uniform time clustering")
        # initialize cluster centers
        self.clustercenters = np.zeros(
            (self.k, self.data_producer.dimension()), dtype=np.float32)

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        """

        :param X:
            coordinates. axis 0: time, axes 1-..: coordinates
        :param itraj:
            index of the current trajectory
        :param t:
            time index of first frame within trajectory
        :param first_chunk:
            boolean. True if this is the first chunk globally.
        :param last_chunk_in_traj:
            boolean. True if this is the last chunk within the trajectory.
        :param last_chunk:
            boolean. True if this is the last chunk globally.
        :param _ipass:
            number of pass through data
        :param Y:
            time-lagged data (if available)
        :return:
        """
        L = np.shape(X)[0]
        if ipass == 0:
            # initialize
            if (first_chunk):
                # initialize time counters
                # time in previous trajectories
                self._tprev = 0
                # number of clusters yet
                self._n = 0
                # time segment length between cluster centers
                self._dt = self.data_producer.n_frames_total(stride=stride) / self.k
                # first data point in the middle of the time segment
                self._nextt = self._dt / 2
            # final time we can go to with this chunk
            maxt = self._tprev + t + L
            # harvest cluster centers from this chunk until we have left it
            while (self._nextt < maxt and self._n < self.k):
                i = self._nextt - self._tprev - t
                self.clustercenters[self._n] = X[i]
                self._n += 1
                self._nextt += self._dt
            if last_chunk_in_traj:
                self._tprev += self.data_producer.trajectory_length(itraj, stride=stride)
        if ipass == 1:
            # discretize all
            if t == 0:
                n = self.data_producer.trajectory_length(itraj, stride=stride)
                self.dtrajs.append(np.zeros(n, dtype=int))
            self.dtrajs[itraj][t:t+L] = self.map(X)
            if last_chunk:
                return True  # done!

        return False  # not done yet.

    def _map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def get_discrete_trajectories(self):
        return self.dtrajs
