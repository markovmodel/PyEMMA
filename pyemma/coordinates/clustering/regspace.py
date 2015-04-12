'''
Created on 26.01.2015

@author: marscher
'''

from pyemma.util.annotators import doc_inherit
from pyemma.coordinates.clustering.interface import AbstractClustering

import regspatial

import numpy as np
import warnings

__all__ = ['RegularSpaceClustering']


class RegularSpaceClustering(AbstractClustering):

    """Clusters data objects in such a way, that cluster centers are at least in
    distance of dmin to each other according to the given metric.
    The assignment of data objects to cluster centers is performed by
    Voronoi partioning. That means, that a data object is assigned to
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

        self._dmin = dmin
        # temporary list to store cluster centers
        self._clustercenters = []
        self.max_clusters = max_clusters

    @doc_inherit
    def describe(self):
        return "[RegularSpaceClustering dmin=%i]" % self._dmin

    @property
    def dmin(self):
        return self._dmin

    @dmin.setter
    def dmin(self, d):
        if d < 0:
            raise ValueError("d has to be positive")

        self._dmin = float(d)
        self._parametrized = False

    def _map_to_memory(self):
        # nothing to do, because memory-mapping of the discrete trajectories is
        # done in parametrize
        pass

    def _get_memory_per_frame(self):
        # 4 bytes per frame for an integer index
        return 4

    def _get_constant_memory(self):
        # memory for cluster centers and discrete trajectories
        return 4 * self.data_producer.dimension() + 4 * self.data_producer.n_frames_total()

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        """
        first pass: calculate clustercenters
         1. choose first datapoint as centroid
         2. for all X: calc distances to all clustercenters
         3. add new centroid, if min(distance to all other clustercenters) >= dmin
        second pass: assign data to discrete trajectories
        """
        if ipass == 0:
            try:
                regspatial.cluster(X.astype(np.float32, order='C', copy=False),
                                   self._clustercenters, self._dmin,
                                   self.metric, self.max_clusters)
                # finished regularly
                if last_chunk:
                    self.clustercenters = np.array(self._clustercenters)
                    return True  # finished!
            except RuntimeError:
                msg = 'Maximum number of cluster centers reached.' \
                      ' Consider increasing max_clusters or choose' \
                      ' a larger minimum distance, dmin.'
                self._logger.warning(msg)
                warnings.warn(msg)
                # finished anyway, because we have no more space for clusters. Rest of trajectory has no effect
                self.clustercenters = np.array(self._clustercenters)
                return True

        return False

    def _param_finish(self):
        del self._clustercenters  # delete temporary
