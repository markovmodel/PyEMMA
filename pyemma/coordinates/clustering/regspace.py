'''
Created on 26.01.2015

@author: marscher
'''

from pyemma.util.log import getLogger
from pyemma.util.annotators import doc_inherit
from pyemma.coordinates.clustering.interface import AbstractClustering

import regspatial

import numpy as np
import warnings

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

    def __init__(self, dmin, max_clusters=1000):
        super(RegularSpaceClustering, self).__init__()

        self.dmin = dmin
        # temporary list to store cluster centers
        self._clustercenters = []
        self.max_clusters = max_clusters

    @doc_inherit
    def describe(self):
        return "[RegularSpaceClustering dmin=%i]" % self.dmin

    @doc_inherit
    def _map_to_memory(self):
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
            try:
                regspatial.cluster(X.astype(np.float32,order='C',copy=False), self._clustercenters, self.dmin, self.metric, self.max_clusters)
            except RuntimeError as e:
                msg = 'Maximum number of cluster centers reached.' \
                      ' Consider increasing max_clusters or choose' \
                      ' a larger minimum distance, dmin.'
                log.warning(msg)
                warnings.warn(msg)
                return False

        elif ipass == 1:
            # discretize all
            if t == 0:
                if itraj == 0:
                    log.debug("mk array")
                    assert len(self._clustercenters) >= 1
                    # create numpy array from clustercenters list
                    self.clustercenters = np.array(self._clustercenters)

                log.debug("shape of clustercenters: %s" %
                          str(self.clustercenters.shape))
                log.info("number of clustercenters: %i" %
                         len(self.clustercenters))
                n = self.data_producer.trajectory_length(itraj)
                self.dtrajs.append(np.empty(n, dtype=np.int64))
            L = np.shape(X)[0]
            
            self.dtrajs[itraj][t:t+L] = self.map(X)
            if last_chunk:
                return True  # finished!

        return False

    def param_finish(self):
        del self._clustercenters  # delete temporary
