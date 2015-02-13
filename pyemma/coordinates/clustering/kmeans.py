'''
Created on 22.01.2015

@author: marscher
'''
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ..transform.transformer import Transformer

from pyemma.util.log import getLogger

log = getLogger('KmeansClustering')

__all__ = ['KmeansClustering']


class KmeansClustering(Transformer):

    r"""
    Kmeans clustering

    Parameters
    ----------
    n_clusters : int
        amount of cluster centers
    max_iter : int 
        how many iterations per chunk?
    """

    def __init__(self, n_clusters, max_iter=1000):
        super(KmeansClustering, self).__init__()
        # TODO: if we do not set a random_state here (eg. a forced seed) we get slightly different cluster centers each run
        self.algo = MiniBatchKMeans(n_clusters,
                                    max_iter=max_iter,
                                    batch_size=self.chunksize,
                                    verbose=True,
                                    )

        self.dtrajs = []

# TODO: make changes to chunksize/batchsize possible
#     @property
#     def chunksize(self):
#         return self._chunksize
# 
#     @chunksize.setter
#     def chunksize(self, cs):
#         self._chunksize = cs
#         self.algo.set_params(batchsize=cs)

    def dimension(self):
        return 1

    def get_memory_per_frame(self):
        return 1

    def get_constant_memory(self):
        return 1

    def _ensure2d(self, X):
        X = X.reshape((-1, 1))
        log.debug("new shape: %s" % str(X.shape))
        return X

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                       last_chunk, ipass, Y=None):
        if X.ndim == 1:
            X = self._ensure2d(X)

        if ipass == 0:
            self.algo.partial_fit(X)
        if ipass == 1:
            # discretize all
            if t == 0:
                n = self.data_producer.trajectory_length(itraj)
                self.dtrajs.append(np.empty(n))
            assignment = self.algo.predict(X)
            self.dtrajs[itraj][:] = assignment

            if last_chunk:
                return True

    def map(self, X):
        if X.ndim == 1:
            X = self._ensure2d(X)
        d = self.algo.predict(X)
        return d
