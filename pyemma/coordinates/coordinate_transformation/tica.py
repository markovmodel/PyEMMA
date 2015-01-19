'''
Created on 19.01.2015

@author: marscher
'''
import numpy as np
from transformer import Transformer


class TICA(Transformer):

    '''
    classdocs
    '''

    def __init__(self, lag):
        '''
        Constructor
        '''
        super(TICA, self).__init__()
        self.lag = lag
        self.cov = None
        self.cov_tau = None
        # mean
        self.mu = None

    def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        """
        :param X:
            coordinates. axis 0: time, axes 1-..: coordinates
        :param itraj:
            index of the current trajectory
        :param t:
            time index of first frame within trajectory
        :param first_chunk:
            boolean. True if this is the first chunk globally.
        :param last_chunk_in_traj:
            boolean. True if this is the last chunk within the trajectory.
        :param last_chunk:
            boolean. True if this is the last chunk globally.
        :param ipass:
            number of pass through data
        :param Y:
            time-lagged data (if available)
        :return:
       """
        if ipass == 0:
            if first_chunk:
                self.mu = np.zeros(self.data_producer.dimension())
                self.mu += np.sum(X, axis=0)
                self.N += np.shape(X)[0]
            if last_chunk:
                self.mu /= self.N

        if ipass == 1:
            if first_chunk:
                dim = self.data_producer.dimension()
                self.cov = np.zeros((dim, dim))
                self.cov_tau = np.zeros_like(self.cov)

            assert Y is not None, "time lagged input missing"

            X_meanfree = X - self.mu
            Y_meanfree = Y - self.mu
            self.cov += np.dot(X_meanfree.T, X_meanfree)
            self.cov_tau += np.dot(X_meanfree.T, Y_meanfree)

            if last_chunk:
                self.cov /= self.N
                self.cov_tau /= self.N
                
                # diagonalize
                # PCA of input paramters
                sigma2PC, W = np.linalg.eig(self.cov)
                sigmaPC=np.array(np.sqrt(sigma2PC),ndmin=2)
