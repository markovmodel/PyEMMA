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

    def __init__(self, data_producer, lag, output_dimension, symmetrize=True):
        '''
        Constructor
        '''
        super(TICA, self).__init__()

        self.data_producer = data_producer
        self.lag = lag
        self.output_dim = output_dimension

        self.symmetrize = symmetrize

        # covariances
        self.cov = None
        self.cov_tau = None
        # mean
        self.mu = None
        self.U = None
        self.lambdas = None
        self.N = 0

        self.parameterized = False

    def get_lag(self):
        return self.lag

    def describe(self):
        return "TICA, lag = %s output dimension = %s" \
            % (self.lag, self.output_dimension())

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self.output_dim

    def output_dimension(self):
        return self.output_dim

    def get_constant_memory(self):
        # TODO: change me
        return self.data_producer.dimension() ** 2

    def parametrization_finished(self):
        return self.parameterized

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
            # TODO: use a more advanced algo for mean calculation
            if first_chunk:
                self.mu = np.zeros(self.data_producer.dimension())

            self.mu += np.sum(X, axis=0)
            self.N += np.shape(X)[0]

            if last_chunk:
                self.mu /= self.N
                print "mean:\n", self.mu

        if ipass == 1:
            if first_chunk:
                dim = self.data_producer.dimension()
                assert dim > 0, "zero dimension from data producer"
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

                if self.symmetrize:
                    self.cov_tau = (self.cov_tau + self.cov_tau.T) / 2.

                self.U, self.lambdas = self._diagonalize()
                self.parameterized = True

    def _diagonalize(self):

        print "covariance:\n", self.cov
        # diagonalize covariance matrices
        sigma2PC, W = np.linalg.eig(self.cov)
        sigmaPC = np.array(np.sqrt(sigma2PC), ndmin=2)

        CovtauPC = np.empty_like(self.cov_tau)
        CorrtauPC = np.empty_like(CovtauPC)

        # Rotate the CovtauIP to the basis of PCs
        CovtauPC = np.dot(W.T, np.dot(self.cov_tau, W))

        # Symmetrize CovtauPC
        CovtauPC = (CovtauPC + CovtauPC.T) / 2.

        # Transform lagged covariance of PCs in lagged correlation of PCs
        CorrtauPC = CovtauPC / sigmaPC.T.dot(sigmaPC)

        # Diagonalize CorrtauPC
        lambdas, V_tau = np.linalg.eig(CorrtauPC)

        # Sort according to absolute value of lambda_tau
        abs_order_descend = np.argsort(np.abs(lambdas))[::-1]
        lambdas = lambdas[abs_order_descend]
        V_tau = V_tau[:, abs_order_descend]

        # Compute U
        U = np.dot(
            W, np.dot(np.diag(1 / sigmaPC.squeeze()), V_tau))

        return U, lambdas

    def map(self, X):
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.U[:, 0:self.output_dim])
        return Y
