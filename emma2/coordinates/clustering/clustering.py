'''
Created on Jan 5, 2014

@author: noe
'''

import emma2.util.pystallone as stallone

class Clustering:
    """
    Wrapper to stallone clustering
    """
    _jclustering = None
    
    def __init__(self, jclustering):
        self._jclustering = jclustering
    
    def __jclustering(self):
        return _jclustering
    
    def nclusters(self):
        """
        Returns the number of clusters
        """
        return self._jclustering.getNumberOfClusters()
    
    def clustercenters(self):
        return stallone.stallone_array_to_ndarray(self._jclustering.getClusterCenters())
    
    def clusterindexes(self):
        return stallone.stallone_array_to_ndarray(self._jclustering.getClusterIndexes())
    
    def assign(self, x):
        return self._jclustering.getClusterAssignment().assign(stallone.ndarray_to_stallone_array(x))
