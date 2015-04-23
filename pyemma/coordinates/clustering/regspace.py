
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
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
    Voronoi partioning.

    Regular space clustering [1]_ is very similar to Hartigan's leader algorithm [2]_. It consists of two passes through
    the data. Initially, the first data point is added to the list of centers. For every subsequent data point, if
    it has a greater distance than dmin from every center, it also becomes a center. In the second pass, a Voronoi
    discretization with the computed centers is used to partition the data.


    Parameters
    ----------
    dmin : float
        minimum distance between all clusters.
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

     References
    ----------
    .. [1] Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD, Schuette Ch and Noe F. 2011.
        Markov models of molecular kinetics: Generation and Validation.
        J. Chem. Phys. 134, 174105.
    .. [2] Hartigan J. Clustering algorithms.
        New York: Wiley; 1975.

    """

    def __init__(self, dmin, max_clusters=1000, metric='euclidean'):
        super(RegularSpaceClustering, self).__init__(metric=metric)

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

    def _param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        self._logger.info("Running regular space clustering")

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
                    self.n_clusters = self.clustercenters.shape[0]
                    return True  # finished!
            except RuntimeError:
                msg = 'Maximum number of cluster centers reached.' \
                      ' Consider increasing max_clusters or choose' \
                      ' a larger minimum distance, dmin.'
                self._logger.warning(msg)
                warnings.warn(msg)
                # finished anyway, because we have no more space for clusters. Rest of trajectory has no effect
                self.clustercenters = np.array(self._clustercenters)
                self.n_clusters = self.clustercenters.shape[0]
                return True

        return False

    def _param_finish(self):
        del self._clustercenters  # delete temporary