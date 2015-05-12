# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
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

"""
Created on 22.01.2015

@author: marscher, noe
"""

import kmeans_clustering
import math
import random
import numpy as np

from pyemma.util.annotators import doc_inherit
from pyemma.coordinates.clustering.interface import AbstractClustering

__all__ = ['KmeansClustering']


class KmeansClustering(AbstractClustering):
    r"""
    Kmeans clustering

    Parameters
    ----------
    n_clusters : int
        amount of cluster centers
    max_iter : int 
        how many iterations per chunk?
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')
    tolerance : float
        if the cluster centers' change did not exceed tolerance, stop iterating
    init_strategy : string
        can be either 'kmeans++' or 'uniform', determining how the initial cluster centers are being chosen
    """

    def __init__(self, n_clusters, max_iter=5, metric='euclidean', tolerance=1e-5, init_strategy='kmeans++'):
        super(KmeansClustering, self).__init__(metric=metric)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self._cluster_centers = []
        self._centers_iter_list = []
        self._tolerance = tolerance
        self._in_memory_chunks = []
        self._init_strategy = init_strategy

    def _param_init(self):
        self._cluster_centers = []
        self._init_centers_indices = {}
        if self._init_strategy == 'uniform':
            traj_lengths = self.trajectory_lengths(stride=self._param_with_stride)
            total_length = sum(traj_lengths)
            # gives random samples from each trajectory such that the cluster centers are distributed percentage-wise
            # with respect to the trajectories length
            for idx, traj_len in enumerate(traj_lengths):
                self._init_centers_indices[idx] = random.sample(range(0, traj_len), int(
                    math.ceil((traj_len / float(total_length)) * self.n_clusters)))

    @doc_inherit
    def describe(self):
        return "[Kmeans, k=%i]" % self.n_clusters

    def _get_memory_per_frame(self):
        return 1

    def _get_constant_memory(self):
        return 1

    def _map_to_memory(self):
        # results mapped to memory during parametrize
        pass

    @doc_inherit
    def describe(self):
        return "[Kmeans, k=%i]" % self.n_clusters

    def _param_finish(self):
        self.clustercenters = np.array(self._cluster_centers)
        del self._cluster_centers
        if self._init_strategy == 'uniform':
            del self._centers_iter_list
            del self._init_centers_indices

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        # first pass: gather data and run k-means
        if ipass == 0:
            # beginning - compute
            if first_chunk:
                mem_req = 1e-6 * 2 * X[0, :].nbytes * self.n_frames_total(stride=stride)
                self._logger.warn('K-means implementation is currently memory inefficient.'
                                  ' This calculation needs %i megabytes of main memory.'
                                  ' If you get a memory error, try using a larger stride.'
                                  % mem_req)

            # appends a true copy
            self._in_memory_chunks.append(X[:, :])

            # initialize uniform cluster centers
            if self._init_strategy == 'uniform':
                if itraj in self._init_centers_indices.keys():
                    for l in xrange(len(X)):
                        if len(self._cluster_centers) < self.n_clusters and t + l in self._init_centers_indices[itraj]:
                            self._cluster_centers.append(X[l].astype(np.float32, order='C'))

            # run k-means in the end
            if last_chunk:
                # concatenate all data
                all_data = np.vstack(self._in_memory_chunks)
                # free part of the memory
                del self._in_memory_chunks

                if self._init_strategy == 'kmeans++':
                    cc = kmeans_clustering.init_centers(all_data.astype(np.float32, order='C'),
                                                        self.metric, self.n_clusters)
                    self._cluster_centers = [c for c in cc]
                # run k-means with all the data
                self._logger.info("Accumulated all data, running kmeans on " + str(all_data.shape))
                it = 0
                while it < self.max_iter:
                    old_centers = self._cluster_centers
                    self._cluster_centers = kmeans_clustering.cluster(all_data.astype(np.float32, order='C'),
                                                                      self._cluster_centers, self.metric)
                    self._cluster_centers = [row for row in self._cluster_centers]
                    if np.allclose(old_centers, self._cluster_centers, rtol=self._tolerance):
                        break
                    it += 1

            # done
            if last_chunk:
                return True
        return True