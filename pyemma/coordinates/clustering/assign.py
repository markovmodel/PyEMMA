from __future__ import absolute_import
from pyemma.util import types
import six

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
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.clustering.interface import AbstractClustering
import numpy as np


class AssignCenters(AbstractClustering):

    """Assigns given (pre-calculated) cluster centers. If you already have
    cluster centers from somewhere, you use this class to assign your data to it.

    Parameters
    ----------
    clustercenters : path to file (csv) or npyfile or ndarray
        cluster centers to use in assignment of data

    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')


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

    def __init__(self, clustercenters, metric='euclidean'):
        super(AssignCenters, self).__init__(metric=metric)

        if isinstance(clustercenters, six.string_types):
            from pyemma.coordinates.data import create_file_reader
            reader = create_file_reader(clustercenters, None, None)
            self._clustercenters = reader.get_output()[0]
        else:
            self._clustercenters = np.array(clustercenters, dtype=np.float32, order='C')

        # sanity check.
        if not self.clustercenters.ndim == 2:
            raise ValueError('cluster centers have to be 2d')

        # since we provided centers, this transformer is already parametrized.
        self._parametrized = True

    def describe(self):
        return "[AssignCenters c=%s]" % self.clustercenters

    @AbstractClustering.data_producer.setter
    def data_producer(self, dp):
        # check dimensions
        dim = self.clustercenters.shape[1]
        if not dim == dp.dimension():
            raise ValueError('cluster centers have wrong dimension. Have dim=%i'
                             ', but input has %i' % (dim, dp.dimension()))
        AbstractClustering.data_producer.fset(self, dp)

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                        last_chunk, ipass, Y=None, stride=1):
        # discretize all
        if t == 0:
            n = self.data_producer.trajectory_length(itraj, stride=stride)
            self._dtrajs.append(np.empty(n, dtype=self.output_type()))

        L = np.shape(X)[0]
        self._dtrajs[itraj][t:t+L] = self._map_array(X).squeeze()

        if last_chunk:
            return True
