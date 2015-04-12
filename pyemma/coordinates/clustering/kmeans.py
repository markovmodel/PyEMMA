'''
Created on 22.01.2015

@author: marscher, noe
'''
import numpy as np
from sklearn.cluster import KMeans as _sklearn_kmeans

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

    """

    def __init__(self, n_clusters, max_iter=5):
        super(KmeansClustering, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self._algo = _sklearn_kmeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        self._chunks = []

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

#     def param_init(self):
#         # ensure we can cluster data
#         input_dim = self.data_producer.dimension()
#         if input_dim < self.n_clusters:
#             raise ValueError("Input dimension lower than number of clusters!")

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        # first pass: gather data and run k-means
        if ipass == 0:
            # beginning - compute
            if first_chunk:
                memreq = 1e-6 * 2 * X[0, :].nbytes * self.n_frames_total(stride=stride)
                self._logger.warn('K-means implementation is currently memory inefficient.'
                                  ' This calculation needs %i megabytes of main memory.'
                                  ' If you get a memory error, try using a larger stride.'
                                  % memreq)

            # appends a true copy
            self._chunks.append(X[:, :])

            # run k-means in the end
            if last_chunk:
                # concatenate all data
                alldata = np.vstack(self._chunks)
                # free part of the memory
                del self._chunks
                # run k-means with all the data
                self._logger.info("Pass 1: Accumulated all data, running kmeans on "+str(alldata.shape))
                self._algo.fit(alldata)

        # second pass: assign states
        if ipass == 1:
            if first_chunk:
                self._logger.info("Pass 2: Assigning data")
            # discretize all
            if t == 0:
                n = self.data_producer.trajectory_length(itraj, stride=stride)
                self.dtrajs.append(np.empty(n, dtype=int))

            assignment = self._algo.predict(X)
            self.dtrajs[itraj][t: t + assignment.shape[0]] = assignment

            # done
            if last_chunk:
                return True

    def _param_finish(self):
        self.clustercenters = self._algo.cluster_centers_

    def _map_array(self, X):
        if X.ndim == 1:
            X = self._ensure2d(X)
        d = self._algo.predict(X)
        return d[:,None] # always return a column vector in this function
