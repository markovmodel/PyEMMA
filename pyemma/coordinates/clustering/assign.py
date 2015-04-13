'''
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.clustering.interface import AbstractClustering
import numpy as np


class AssignCenters(AbstractClustering):

    """Assigns given (precalculated) cluster centers. If you already have
    cluster centers from somewhere, you use this class to assign your data to it.

    Parameters
    ----------
    clustercenters : path to file (csv) or ndarray
        cluster centers to use in assignment of data

    Examples
    --------
    Assuming you have stored your centers in a CSV file:

    >>> from pyemma.coordinates.clustering import AssignCenters
    >>> from pyemma.coordinates import discretizer
    >>> reader = ...
    >>> assign = AssignCenters('my_centers.dat')
    >>> disc = discretizer(reader, cluster=assign)
    >>> disc.parametrize()

    """

    def __init__(self, clustercenters):
        super(AssignCenters, self).__init__()

        if isinstance(clustercenters, basestring):
            self.clustercenters = np.loadtxt(clustercenters)

        self.clustercenters = np.array(clustercenters, dtype=np.float32, order='C')

        # since we provided centers, this transformer is already parametrized.
        self._parametrized = True

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
