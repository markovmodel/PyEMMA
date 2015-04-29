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
import warnings
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

    """

    def __init__(self, n_clusters, max_iter=5, metric='euclidean'):
        super(KmeansClustering, self).__init__(metric=metric)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self._clustercenters = []
        self._centers_iter_list = []

    @doc_inherit
    def describe(self):
        return "[Kmeans, k=%i]" % self.n_clusters

    def _get_memory_per_frame(self):
        return 1

    def _get_constant_memory(self):
        return 1

    def _map_to_memory(self):
        # resutls mapped to memory during parameterize
        pass

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        # first pass: gather data and run k-means
        if ipass == 0 and first_chunk:
            # initialize cluster centers
            self._clustercenters = [
                np.array(X[np.random.randint(0, len(X))], dtype=np.float32) for _ in xrange(self.n_clusters)
            ]

        # allocate assigns
        data_assigns = [0] * len(X)

        try:
            #print "cctype=%s, assignstype=%s, metrictype=%s" % (type(self._clustercenters), type(data_assigns), type(self.metric))
            chunk = X.astype(np.float32, order='C', copy=False)
            #print "Feeding chunk with len=%s, type=%s, chunk=%s"%(len(chunk), type(chunk), self._clustercenters)
            new_centers = kmeans_clustering.cluster(chunk,
                                                    self._clustercenters, data_assigns,
                                                    self.metric)
            self._centers_iter_list.append(new_centers)
        except RuntimeError:
            msg = 'Maximum number of cluster centers reached.' \
                  ' Consider increasing max_clusters or choose' \
                  ' a larger minimum distance, dmin.'
            self._logger.warning(msg)
            warnings.warn(msg)
            return True

        # run k-means in the end
        done = ipass + 1 >= self.max_iter
        if last_chunk:
            #center_sum = np.zeros([self.n_clusters, len(X[0])])
            center_sum = [np.zeros([len(X[0])], dtype=np.float32) for _ in range(0, self.n_clusters)]
            for chunk_center in self._centers_iter_list:
                #print "cc=%s"%chunk_center
                for idx, chunk_center_element in enumerate(chunk_center):
                    center_sum[idx] += chunk_center_element

            old_centers = self._clustercenters
            n_centers_in_iter_list = len(self._centers_iter_list)
            self._clustercenters = [center / n_centers_in_iter_list for center in center_sum]
            self._centers_iter_list = []
            if np.allclose(old_centers, self._clustercenters, rtol=1e-5):
                done = True
        return done

    def _param_finish(self):
        self.clustercenters = np.array(self._clustercenters)
        del self._clustercenters
        del self._centers_iter_list