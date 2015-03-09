__author__ = 'noe'

from pyemma.util.log import getLogger
from pyemma.coordinates.clustering.interface import AbstractClustering

import numpy as np

log = getLogger('UniformTimeClustering')
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

    def dimension(self):
        return 1

    def get_memory_per_frame(self):
        """
        Returns the memory requirements per frame, in bytes

        :return:
        """
        # 4 bytes per frame for an integer index
        return 0

    def get_constant_memory(self):
        """
        Returns the constant memory requirements, in bytes

        :return:
        """
        # memory for cluster centers and discrete trajectories
        return self.k * 4 * self.data_producer.dimension() + 4 * self.data_producer.n_frames_total()

    def param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        log.info("Running uniform time clustering")
        # initialize cluster centers
        self.clustercenters = np.zeros(
            (self.k, self.data_producer.dimension()), dtype=np.float32)
        # initialize time counters
        self.tprev = 0
        self.ipass = 0
        self.n = 0
        self.stride = self.data_producer.n_frames_total() / self.k
        self.nextt = self.stride / 2

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
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
        :param ipass:
            number of pass through data
        :param Y:
            time-lagged data (if available)
        :return:
        """
        log.debug("itraj = " + str(itraj) + ". t = " + str(t) + ". last_chunk_in_traj = " + str(last_chunk_in_traj)
                  + " last_chunk = " + str(last_chunk) + " ipass = " + str(ipass))

        L = np.shape(X)[0]
        if ipass == 0:
            maxt = self.tprev + t + L
            while (self.nextt < maxt):
                i = self.nextt - self.tprev - t
                # FIXME: n out of bounds
                self.clustercenters[self.n] = X[i]
                self.n += 1
                self.nextt += self.stride
            if last_chunk_in_traj:
                self.tprev += self.data_producer.trajectory_length(itraj)
        if ipass == 1:
            # discretize all
            if t == 0:
                n = self.data_producer.trajectory_length(itraj)
                self.dtrajs.append(np.zeros(n, dtype=int))
            for i in xrange(L):
                self.dtrajs[itraj][i + t] = self.map(X[i])
            if last_chunk:
                return True  # done!

        return False  # not done yet.

    def map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def get_discrete_trajectories(self):
        return self.dtrajs
