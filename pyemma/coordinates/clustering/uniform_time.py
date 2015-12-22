
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import absolute_import, division

import math

from pyemma.coordinates.clustering.interface import AbstractClustering

import numpy as np


__author__ = 'noe'
__all__ = ['UniformTimeClustering']


class UniformTimeClustering(AbstractClustering):
    r"""Uniform time clustering"""

    def __init__(self, n_clusters=2, metric='euclidean', stride=1):
        """r
        Uniform time clustering

        Parameters
        ----------
        n_clusters : int
            amount of desired cluster centers
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')
        """
        super(UniformTimeClustering, self).__init__(metric=metric)
        self.set_params(n_clusters=n_clusters, metric=metric, stride=stride)

    def describe(self):
        return "[Uniform time clustering, k = %i, inp_dim=%i]" \
                % (self.n_clusters, self.data_producer.dimension())

    def _param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        # initialize cluster centers
        if not self.n_clusters:
            traj_lengths = self.trajectory_lengths(stride=self._param_with_stride)
            total_length = sum(traj_lengths)
            self.n_clusters = min(int(math.sqrt(total_length)), 5000)
            self._logger.info("The number of cluster centers was not specified, "
                              "using min(sqrt(N), 5000)=%s as n_clusters." % self.n_clusters)

        self._clustercenters = np.zeros(
            (self.n_clusters, self.data_producer.dimension()), dtype=np.float32)

    def _estimate(self, iterable, **kw):

        stride = kw['stride'] if 'stride' in kw else self.stride

        # initialize time counters
        T = iterable.n_frames_total(stride=stride)
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
        self._dt = T // self.n_clusters
        # first data point in the middle of the time segment
        self._nextt = self._dt // 2

        last_itraj = -1
        t = 0
        iterator = iterable.iterator(stride=self.stride, return_trajindex=True, **kw)

        for itraj, X in iterator:

            if itraj != last_itraj:
                if last_itraj != -1:
                    self._tprev += self.trajectory_length(itraj=last_itraj, stride=stride)
                last_itraj = itraj
                t = 0

            L = np.shape(X)[0]
            t += L

            # final time we can go to with this chunk
            maxt = self._tprev + t + L
            # harvest cluster centers from this chunk until we have left it
            while (self._nextt < maxt and self._n < self.n_clusters):
                i = self._nextt - self._tprev - t
                self._clustercenters[self._n] = X[i]
                self._n += 1
                self._nextt += self._dt
