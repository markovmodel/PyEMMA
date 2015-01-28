'''
Created on 26.01.2015

@author: marscher
'''

from pyemma.util.log import getLogger

from transformer import Transformer
import numpy as np


log = getLogger('RegSpaceClustering')


class RegularSpaceClustering(Transformer):

    """Clusters data objects in such a way, that cluster centers are at least in
    distance of dmin to eachother according to the given metric.
    The assignment of data objects to cluster centers is performed by
    Voronoi paritioning. That means, that a data object is assigned to
    that clusters center, which has the least distance.

    Parameters
    ----------
    dmin : float

    """

    def __init__(self, dmin):
        super(RegularSpaceClustering, self).__init__()

        self._dmin = dmin
        self._dtrajs = []

        # TODO: determine if list or np array is more efficient.
        self._centroids = []

    def describe(self):
        return "[RegularSpaceClustering dmin=%i]" % self._dmin

    @property
    def dmin(self):
        return self._dmin

    def map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def _distances(self, X):
        """ calculate distance for each frame in X to current list of centroids"""
        dists = np.empty(len(self._centroids))
        # TODO: optimize
        d = X.shape[0]
        for ii, center in enumerate(self._centroids):
            for jj in xrange(d):
                #assert center.shape == X[jj].shape
                dists[ii] = np.linalg.norm(X[jj] - center, 2)
        return dists

    #def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):

        """
        first pass: calculate centroids
        second pass: assign data to discrete trajectories
#         1. choose first datapoint as centroid
#         2. for all X: calc distances to all centroids
#         3. assign 
        """
        if ipass == 0:
            # add first point as first centroid
            if first_chunk:
                self._centroids.append(X[0])

            dists = self._distances(X)
            minIndex = np.argmin(dists)

            # minimal distance of current batch bigger than minimal distance?
            if dists[minIndex] > self._dmin:
                log.debug("dist= %f" % dists[minIndex])
                #log.debug('adding new centroid %s' % X[minIndex])
                self._centroids.append(X[minIndex])

        elif ipass == 1:
            # discretize all
            if t == 0:
                self._dtrajs.append(
                    np.empty(self.data_producer.trajectory_length(itraj)))
            L = np.shape(X)[0]
            # TODO: optimize: assign one chunk at once
            for i in xrange(L):
                self._dtrajs[itraj][i + t] = self.map(X[i])
            if last_chunk:
                return True  # finished!

        return False

    def map(self, x):
        """gets index of closest cluster.
        TODO: If X is a chunk [shape = (n, d)], return an array with all indices.
        """
        dists = self.data_producer.distances(x, self._centroids)
        return np.argmin(dists)

    @property
    def dtrajs(self):
        return self._dtrajs
