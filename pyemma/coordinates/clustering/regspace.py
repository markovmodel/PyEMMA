
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


import warnings

from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.util.annotators import fix_docs
from pyemma.util.exceptions import NotConvergedWarning

import numpy as np
import deeptime as dt

__all__ = ['RegularSpaceClustering']


@fix_docs
class RegularSpaceClustering(AbstractClustering):
    r"""Regular space clustering"""
    __serialize_version = 0

    def __init__(self, dmin, max_centers=1000, metric='euclidean', stride=1, n_jobs=None, skip=0):
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

        from ._ext import RMSDMetric
        dt.clustering.metrics.register("minRMSD", RMSDMetric)

        self._converged = False
        self.set_params(dmin=dmin, metric=metric,
                        max_centers=max_centers, stride=stride, skip=skip)

    def describe(self):
        return "[RegularSpaceClustering dmin=%f, inp_dim=%i]" % (self._dmin, self.data_producer.dimension())

    @property
    def dmin(self):
        """Minimum distance between cluster centers."""
        return self._dmin

    @dmin.setter
    def dmin(self, d):
        d = float(d)
        if d < 0:
            raise ValueError("d has to be positive")

        self._dmin = d

    @property
    def max_centers(self):
        """
        Cutoff during clustering. If reached no more data is taken into account.
        You might then consider a larger value or a larger dmin value.
        """
        return self._max_centers

    @max_centers.setter
    def max_centers(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("max_centers has to be positive")

        self._max_centers = value

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
        used_frames = 0
        regspace = dt.clustering.RegularSpace(dmin=self.dmin, max_centers=self.max_centers,
                                              metric=self.metric, n_jobs=self.n_jobs)

        # from ._ext import regspace
        it = iterable.iterator(return_trajindex=False, stride=self.stride,
                               chunk=self.chunksize, skip=self.skip)
        try:
            with it:
                for X in it:
                    regspace.partial_fit(X.astype(np.float32, order='C', copy=False), n_jobs=self.n_jobs)
                    used_frames += len(X)
            self._converged = True
        except regspace.MaxCentersReachedException:
            self._converged = False
            msg = 'Maximum number of cluster centers reached.' \
                  ' Consider increasing max_centers or choose' \
                  ' a larger minimum distance, dmin.'
            self.logger.warning(msg)
            warnings.warn(msg)
            # pass amount of processed data
            used_data = used_frames / float(it.n_frames_total()) * 100.0
            raise NotConvergedWarning("Used data for centers: %.2f%%" % used_data)
        finally:
            # even if not converged, we store the found centers.
            model = regspace.fetch_model()
            clustercenters = model.cluster_centers.squeeze().reshape(-1, iterable.ndim)
            self._inst = dt.clustering.ClusterModel(clustercenters, metric=self.metric)
            from types import MethodType

            def _assign(self, data, _, n_jobs):
                out = self.transform(data, n_jobs=n_jobs)
                return out

            self._inst.assign = MethodType(_assign, self._inst)
            self.update_model_params(clustercenters=clustercenters,
                                     n_clusters=len(clustercenters))

            if len(clustercenters) == 1:
                self.logger.warning('Have found only one center according to '
                                     'minimum distance requirement of %f' % self.dmin)

        return self
