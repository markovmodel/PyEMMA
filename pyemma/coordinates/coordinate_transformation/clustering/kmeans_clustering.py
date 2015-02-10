'''
Created on 22.01.2015

@author: marscher
'''
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ..transform.transformer import Transformer

__all__ = ['KmeansClustering']


class KmeansClustering(Transformer):

    '''
    classdocs
    '''

    def __init__(self, n_clusters, max_iter=1000):
        '''
        Constructor
        '''
        super(KmeansClustering, self).__init__()

        self.algo = MiniBatchKMeans(n_clusters,
                                    max_iter=max_iter,
                                    batch_size=self.chunksize)

        self.dtrajs = []

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        L = np.shape(X)[0]
        if ipass == 0:
            self.algo.partial_fit(X)
        if ipass == 1:
            # discretize all
            if t == 0:
                self.dtrajs.append(
                    np.empty(self.data_producer.trajectory_length(itraj)))
            for i in xrange(L):
                self.dtrajs[itraj][i + t] = self.map(X[i])

            if last_chunk:
                return True

    def map(self, X):
        d = self.algo.transform(X)
        return np.argmin(d)

