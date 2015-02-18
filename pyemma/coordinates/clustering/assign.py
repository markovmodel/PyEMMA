'''
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.msm.io import read_matrix
import numpy as np


class AssignCenters(AbstractClustering):

    """
    If you already have cluster centers from somewhere, you use this class to
    assign your data to it.

    Examples
    --------
    Assuming you have stored your centers in a CSV file:
    >>> from pyemma.coordinates.clustering.assign import AssignCenters
    >>> from pyemma.coordinates import discretizer
    >>> reader = ...
    >>> assign = AssignCenters('my_centers.dat')
    >>> disc = discretizer(reader, cluster=assign)
    >>> disc.run()
    """

    def __init__(self, clustercenters):
        """
        Parameters
        ----------
        clustercenters : path to file (csv) or ndarray
            cluster centers to use in assignment of data
        """
        if isinstance(clustercenters, str):
            self.clustercenters = read_matrix(clustercenters)

        assert isinstance(self.clustercenters, np.ndarray)

        self.clustercenters = clustercenters
