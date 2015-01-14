'''
Created on Jan 5, 2014

@author: noe
'''

import numpy as np
import pyemma.util.pystallone as stallone


class Clustering(object):
    r"""Wrapper to stallone clustering.
    
    """

    def __init__(self, jclustering):
        self._jclustering = jclustering

    def __jclustering(self):
        return self._jclustering

    def nclusters(self):
        r"""Returns the number of clusters.
        
        """
        return self._jclustering.getNumberOfClusters()

    def clustercenter(self, i):
        """
        returns the cluster centers
        """
        jclustercenters = self._jclustering.getClusterCenter(i)
        return stallone.stallone_array_to_ndarray(jclustercenters)

    def clustercenters(self):
        r"""Returns the cluster centers.
        
        """
        x0 = self.clustercenter(0)
        nc = self.nclusters()
        centers = np.ndarray(tuple([nc]) + np.shape(x0))
        for i in range(0,nc):
            centers[i] = self.clustercenter(i)
        return centers

    def clusters(self):
        r"""Returns the cluster indexes of the input data set.
        
        """
        jindexes = self._jclustering.getClusterIndexes()
        return stallone.stallone_array_to_ndarray(jindexes)

    def assign(self, X):
        r"""Assigns point X to a cluster and returns its index.

        Parameters
        ----------
        X : numpy ndarray
            coordinate set to be assigned
            
        """
        jX = stallone.ndarray_to_stallone_array(X)
        return self._jclustering.assign(jX)
