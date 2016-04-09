
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


'''
Created on 26.01.2015

@author: marscher
'''

from __future__ import absolute_import

import warnings

from pyemma.coordinates.clustering import regspatial
from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.util.annotators import doc_inherit
from pyemma.util.exceptions import NotConvergedWarning

import numpy as np


__all__ = ['RegularSpaceClustering']


class RegularSpaceClustering(AbstractClustering):
    r"""Regular space clustering"""

    def __init__(self, dmin, max_centers=1000, metric='euclidean', stride=1, n_jobs=None):
        """Clusters data objects in such a way, that cluster centers are at least in
        distance of dmin to each other according to the given metric.
        The assignment of data objects to cluster centers is performed by
        Voronoi partioning.

        Regular space clustering [Prinz_2011]_ is very similar to Hartigan's leader
        algorithm [Hartigan_1975]_. It consists of two passes through
        the data. Initially, the first data point is added to the list of centers.
        For every subsequent data point, if it has a greater distance than dmin from
        every center, it also becomes a center. In the second pass, a Voronoi
        discretization with the computed centers is used to partition the data.


        Parameters
        ----------
        dmin : float
            minimum distance between all clusters.
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')
        max_centers : int
            if this cutoff is hit during finding the centers,
            the algorithm will abort.
        n_jobs : int or None, default None
            Number of threads to use during assignment of the data.
            If None, all available CPUs will be used.

        References
        ----------

        .. [Prinz_2011] Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD, Schuette Ch and Noe F. 2011.
            Markov models of molecular kinetics: Generation and Validation.
            J. Chem. Phys. 134, 174105.
        .. [Hartigan_1975] Hartigan J. Clustering algorithms.
            New York: Wiley; 1975.

        """
        super(RegularSpaceClustering, self).__init__(metric=metric, n_jobs=n_jobs)

        self.set_params(dmin=dmin, metric=metric, max_centers=max_centers, stride=stride)

    @doc_inherit
    def describe(self):
        return "[RegularSpaceClustering dmin=%f, inp_dim=%i]" % (self._dmin, self.data_producer.dimension())

    @property
    def dmin(self):
        """Minimum distance between cluster centers."""
        return self._dmin

    @dmin.setter
    def dmin(self, d):
        if d < 0:
            raise ValueError("d has to be positive")

        self._dmin = float(d)
        self._estimated = False

    @property
    def max_centers(self):
        """
        Cutoff during clustering. If reached no more data is taken into account.
        You might then consider a larger value or a larger dmin value.
        """
        return self._max_centers

    @max_centers.setter
    def max_centers(self, value):
        if value < 0:
            raise ValueError("max_centers has to be positive")

        self._max_centers = int(value)
        self._estimated = False

    @property
    def n_clusters(self):
        return self.max_centers

    @n_clusters.setter
    def n_clusters(self, val):
        self.max_centers = val

    def _estimate(self, iterable, **kwargs):
        ########
        # Calculate clustercenters:
        # 1. choose first datapoint as centroid
        # 2. for all X: calc distances to all clustercenters
        # 3. add new centroid, if min(distance to all other clustercenters) >= dmin
        ########
        # temporary list to store cluster centers
        clustercenters = []
        it = iterable.iterator(return_trajindex=False)
        used_frames = 0
        try:
            with iterable.iterator(return_trajindex=False) as it:
                for X in it:
                    used_frames += len(X)
                    regspatial.cluster(X.astype(np.float32, order='C', copy=False),
                                       clustercenters, self.dmin,
                                       self.metric, self.max_centers)
        except RuntimeError:
            msg = 'Maximum number of cluster centers reached.' \
                  ' Consider increasing max_centers or choose' \
                  ' a larger minimum distance, dmin.'
            self._logger.warning(msg)
            warnings.warn(msg)
            # finished anyway, because we have no more space for clusters. Rest of trajectory has no effect
            clustercenters = np.array(clustercenters)
            self.update_model_params(clustercenters=clustercenters,
                                     n_cluster=len(clustercenters))
            # pass amount of processed data
            used_data = used_frames / float(it.n_frames_total()) * 100.0
            raise NotConvergedWarning("Used data for centers: %.2f%%" % used_data)

        clustercenters = np.array(clustercenters)
        self.update_model_params(clustercenters=clustercenters,
                                 n_clusters=len(clustercenters))

        if len(clustercenters) == 1:
            self._logger.warning('Have found only one center according to '
                                 'minimum distance requirement of %f' % self.dmin)

        return self
