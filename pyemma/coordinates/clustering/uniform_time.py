
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

    def __init__(self, n_clusters=2, metric='euclidean', stride=1, n_jobs=None):
        """r
        Uniform time clustering

        Parameters
        ----------
        n_clusters : int
            amount of desired cluster centers. When not specified (None),
            min(sqrt(N), 5000) is chosen as default value,
            where N denotes the number of data points
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')
        stride : int
            stride
        n_jobs : int or None, default None
            Number of threads to use during assignment of the data.
            If None, all available CPUs will be used.
        """
        super(UniformTimeClustering, self).__init__(metric=metric, n_jobs=n_jobs)
        self.set_params(n_clusters=n_clusters, metric=metric, stride=stride)

    def describe(self):
        return "[Uniform time clustering, k = %i, inp_dim=%i]" \
                % (self.n_clusters, self.data_producer.dimension())

    def _estimate(self, iterable, **kw):

        if self.n_clusters is None:
            traj_lengths = self.trajectory_lengths(stride=self.stride)
            total_length = sum(traj_lengths)
            self.n_clusters = min(int(math.sqrt(total_length)), 5000)
            self._logger.info("The number of cluster centers was not specified, "
                              "using min(sqrt(N), 5000)=%s as n_clusters." % self.n_clusters)

        # initialize time counters
        T = iterable.n_frames_total(stride=self.stride)
        if self.n_clusters > T:
            self.n_clusters = T
            self._logger.info('Requested more clusters (k = %i'
                              ' than there are total data points %i)'
                              '. Will do clustering with k = %i'
                              % (self.n_clusters, T, T))

        # first data point in the middle of the time segment
        next_t = (T // self.n_clusters) // 2
        # cumsum of lenghts
        cumsum = np.cumsum(self.trajectory_lengths())
        # distribution of integers, truncate if n_clusters is too large
        linspace = self.stride * np.arange(next_t, T - next_t + 1, (T - 2*next_t + 1) // self.n_clusters)[:self.n_clusters]
        # random access matrix
        ra_stride = np.array([UniformTimeClustering._idx_to_traj_idx(x, cumsum) for x in linspace])
        with iterable.iterator(stride=ra_stride, return_trajindex=False) as it:
            self.clustercenters = np.concatenate([X for X in it])

        assert len(self.clustercenters) == self.n_clusters
        return self

    @staticmethod
    def _idx_to_traj_idx(idx, cumsum):
        prev_len = 0
        for trajIdx, length in enumerate(cumsum):
            if prev_len <= idx < length:
                return trajIdx, idx - prev_len
            prev_len = length
        raise ValueError("Requested index %s was out of bounds [0,%s)" % (idx, cumsum[-1]))
