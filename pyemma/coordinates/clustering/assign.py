
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
Created on 18.02.2015

@author: marscher
'''

from __future__ import absolute_import

import numpy as np
import six

from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.util.annotators import fix_docs


@fix_docs
class AssignCenters(AbstractClustering):

    """Assigns given (pre-calculated) cluster centers. If you already have
    cluster centers from somewhere, you use this class to assign your data to it.

    Parameters
    ----------
    clustercenters : path to file (csv) or npyfile or ndarray
        cluster centers to use in assignment of data
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')
    stride : int
        stride
    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.
    skip : int, default=0
        skip the first initial n frames per trajectory.
    Examples
    --------
    Assuming you have stored your centers in a CSV file:

    >>> from pyemma.coordinates.clustering import AssignCenters
    >>> from pyemma.coordinates import pipeline
    >>> reader = ... # doctest: +SKIP
    >>> assign = AssignCenters('my_centers.dat') # doctest: +SKIP
    >>> disc = pipeline(reader, cluster=assign) # doctest: +SKIP
    >>> disc.parametrize() # doctest: +SKIP

    """

    def __init__(self, clustercenters, metric='euclidean', stride=1, n_jobs=None, skip=0):
        super(AssignCenters, self).__init__(metric=metric, n_jobs=n_jobs)

        if isinstance(clustercenters, six.string_types):
            from pyemma.coordinates.data import create_file_reader
            reader = create_file_reader(clustercenters, None, None)
            clustercenters = reader.get_output()[0]
        else:
            clustercenters = np.array(clustercenters, dtype=np.float32, order='C')

        # sanity check.
        if not clustercenters.ndim == 2:
            raise ValueError('cluster centers have to be 2d')

        self.set_params(clustercenters=clustercenters, metric=metric, stride=stride, skip=skip)

        # since we provided centers, no estimation is required.
        self._estimated = True

    def describe(self):
        return "[{name} centers shape={shape}]".format(name=self.name, shape=self.clustercenters.shape)

    @AbstractClustering.data_producer.setter
    def data_producer(self, dp):
        # check dimensions
        dim = self.clustercenters.shape[1]
        if not dim == dp.dimension():
            raise ValueError('cluster centers have wrong dimension. Have dim=%i'
                             ', but input has %i' % (dim, dp.dimension()))
        AbstractClustering.data_producer.fset(self, dp)

    def _estimate(self, iterable, **kw):
        old_source = self._data_producer
        self.data_producer = iterable
        try:
            self.assign(None, self.stride)
        finally:
            self.data_producer = old_source

        return self
