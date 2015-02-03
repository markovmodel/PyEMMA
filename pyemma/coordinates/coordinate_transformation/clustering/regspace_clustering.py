'''
Created on 26.01.2015

@author: marscher
'''

from pyemma.util.log import getLogger

from pyemma.coordinates.coordinate_transformation.transform.transformer import Transformer
import numpy as np

log = getLogger('RegSpaceClustering')
__all__ = ['RegularSpaceClustering']


class RegularSpaceClustering(Transformer):

    """Clusters data objects in such a way, that cluster centers are at least in
    distance of dmin to each other according to the given metric.
    The assignment of data objects to cluster centers is performed by
    Voronoi paritioning. That means, that a data object is assigned to
    that clusters center, which has the least distance [1] Senne et al.

    Parameters
    ----------
    dmin : float
        minimum distance a new centroid has to have to all other centroids.

     References
    ----------
    .. [1] Senne, Martin, et al. J. Chem Theory Comput. 8.7 (2012): 2223-2238

    """

    def __init__(self, dmin):
        super(RegularSpaceClustering, self).__init__()

        self.dmin = dmin
        self.dtrajs = []

        # TODO: determine if list or np array is more efficient.
        self.centroids = []

    def describe(self):
        return "[RegularSpaceClustering dmin=%i]" % self.dmin

    def map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        """
        first pass: calculate centroids
         1. choose first datapoint as centroid
         2. for all X: calc distances to all centroids
         3. add new centroid, if min(distance to all other centroids) >= dmin
        second pass: assign data to discrete trajectories
        """
        log.debug("t=%i; itraj=%i" % (t, itraj))
        if ipass == 0:
            # add first point as first centroid
            if first_chunk:
                self.centroids.append(X[0])
                log.info("Run regspace clustering with dmin=%f;"
                         " First centroid=%s" % (self.dmin, X[0]))

            for x in X:
                dist = np.fromiter((np.linalg.norm(x - c, 2)
                                    for c in self.centroids), dtype=np.float32)

                minIndex = np.argmin(dist)
                if dist[minIndex] >= self.dmin:
                    self.centroids.append(x)

        elif ipass == 1:
            # discretize all
            if t == 0:
                self.dtrajs.append(
                    np.empty(self.data_producer.trajectory_length(itraj)))
            L = np.shape(X)[0]
            # TODO: optimize: assign one chunk at once
            for i in xrange(L):
                self.dtrajs[itraj][i + t] = self.map(X[i])
            if last_chunk:
                return True  # finished!

        return False

    def param_finish(self):
        assert len(self.centroids) >= 1
        # create numpy array from centroids list
        self.centroids = np.array(self.centroids)
        log.debug("shape of centroids: %s" % str(self.centroids.shape))
        log.info("number of centroids: %i" % len(self.centroids))

    def map(self, x):
        """gets index of closest cluster.
        """
        dists = self.data_producer.distances(x, self.centroids)
        return np.argmin(dists)
