# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        TODO: super unknown parameter
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')
    """

    def __init__(self, k=2, metric='euclidean'):
        super(UniformTimeClustering, self).__init__(metric=metric)
        self.n_clusters = k

    def describe(self):
        return "[Uniform time clustering, k = %i]" % self.n_clusters

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
        return self.n_clusters * 4 * self.data_producer.dimension() + 4 * self.data_producer.n_frames_total()

    def _param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        self._logger.info("Running uniform time clustering")
        # initialize cluster centers
        self.clustercenters = np.zeros(
            (self.n_clusters, self.data_producer.dimension()), dtype=np.float32)

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
                T = self.data_producer.n_frames_total(stride=stride)
                if self.n_clusters > T:
                    self.n_clusters = T
                    self._logger.info('Requested more clusters (k = %i'
                                      ' than there are total data points %i)'
                                      '. Will do clustering with k = %i'
                                      % (self.n_clusters, T, T))

                # time in previous trajectories
                self._tprev = 0
                # number of clusters yet
                self._n = 0
                # time segment length between cluster centers
                self._dt = T / self.n_clusters
                # first data point in the middle of the time segment
                self._nextt = self._dt / 2
            # final time we can go to with this chunk
            maxt = self._tprev + t + L
            # harvest cluster centers from this chunk until we have left it
            while (self._nextt < maxt and self._n < self.n_clusters):
                i = self._nextt - self._tprev - t
                self.clustercenters[self._n] = X[i]
                self._n += 1
                self._nextt += self._dt
            if last_chunk_in_traj:
                self._tprev += self.data_producer.trajectory_length(
                    itraj, stride=stride)
            if last_chunk:
                return True  # done!

        return False  # not done yet.

    def _map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass
