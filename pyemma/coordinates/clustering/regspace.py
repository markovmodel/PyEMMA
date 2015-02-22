'''
Created on 26.01.2015

@author: marscher
'''

from pyemma.util.log import getLogger
from pyemma.util.annotators import doc_inherit
from pyemma.coordinates.clustering.interface import AbstractClustering

import numpy as np

log = getLogger('RegSpaceClustering')
__all__ = ['RegularSpaceClustering']


class RegularSpaceClustering(AbstractClustering):

    """Clusters data objects in such a way, that cluster centers are at least in
    distance of dmin to each other according to the given metric.
    The assignment of data objects to cluster centers is performed by
    Voronoi paritioning. That means, that a data object is assigned to
    that clusters center, which has the least distance [1] Senne et al.

    Parameters
    ----------
    dmin : float
        minimum distance between all clusters.

     References
    ----------
    .. [1] Senne, Martin, et al. J. Chem Theory Comput. 8.7 (2012): 2223-2238

    """

    def __init__(self, dmin):
        super(RegularSpaceClustering, self).__init__()

        self.dmin = dmin
        # temporary list to store cluster centers
        self._clustercenters = []

    @doc_inherit
    def describe(self):
        return "[RegularSpaceClustering dmin=%i]" % self.dmin

    @doc_inherit
    def map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def dimension(self):
        return 1

    @doc_inherit
    def get_memory_per_frame(self):
        # 4 bytes per frame for an integer index
        return 4

    @doc_inherit
    def get_constant_memory(self):
        # memory for cluster centers and discrete trajectories
        return 4 * self.data_producer.dimension() + 4 * self.data_producer.n_frames_total()

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        """
        first pass: calculate clustercenters
         1. choose first datapoint as centroid
         2. for all X: calc distances to all clustercenters
         3. add new centroid, if min(distance to all other clustercenters) >= dmin
        second pass: assign data to discrete trajectories
        """
        log.debug("t=%i; itraj=%i" % (t, itraj))
        if ipass == 0:
            # add first point as first centroid
            if first_chunk:
                self._clustercenters.append(X[0])
                log.info("Run regspace clustering with dmin=%f;"
                         " First centroid=%s" % (self.dmin, X[0]))
            # TODO: optimize with cython, support different metrics
            # see mixtape.libdistance
            for x in X:
                dist = np.fromiter((np.linalg.norm(x - c, 2)
                                    for c in self._clustercenters), dtype=np.float32)

                minIndex = np.argmin(dist)
                if dist[minIndex] >= self.dmin:
                    self._clustercenters.append(x)

        elif ipass == 1:
            # discretize all
            if t == 0:
                assert len(self._clustercenters) >= 1
                # create numpy array from clustercenters list
                self.clustercenters = np.array(self._clustercenters)

                log.debug("shape of clustercenters: %s" %
                          str(self.clustercenters.shape))
                log.info("number of clustercenters: %i" %
                         len(self.clustercenters))
                n = self.data_producer.trajectory_length(itraj)
                self.dtrajs.append(np.empty(n, dtype=int))
            L = np.shape(X)[0]
            # TODO: optimize: assign one chunk at once
            for i in xrange(L):
                self.dtrajs[itraj][i + t] = self.map(X[i])
            if last_chunk:
                return True  # finished!

        return False

    def param_finish(self):
        del self._clustercenters  # delete temporary
