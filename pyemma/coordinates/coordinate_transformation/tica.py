'''
Created on 19.01.2015

@author: marscher
'''
import logging
import numpy as np
from transformer import Transformer


class TICA(Transformer):
    """
    Time-lagged independent component analysis (TICA)

    Given a sequence of multivariate data X_t, computes the mean-free covariance and
    time-lagged covariance matrix:
    C_0 =   (X_t - mu)^T (X_t - mu)
    C_tau = (X_t - mu)^T (X_t+tau - mu)
    and solves the eigenvalue problem
    C_tau r_i = C_0 lambda_i r_i,
    where r_i are the independent compontns and lambda are their respective normalized
    time-autocorrelations. The eigenvalues are related to the relaxation timescale by
    t_i = -tau / ln |lambda_i|

    When used as a dimension reduction method, the input data is projected onto the
    dominant independent components.

    """

    def __init__(self, lag, output_dimension, symmetrize=True):
        '''
        Constructor
        '''
        super(TICA, self).__init__()

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


    def param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        logging.info("Running TICA")
        self.N = 0
        # create mean array and covariance matrices
        self.mu = np.zeros(self.data_producer.dimension())
        dim = self.data_producer.dimension()
        assert dim > 0, "zero dimension from data producer"
        self.cov = np.zeros((dim, dim))
        self.cov_tau = np.zeros_like(self.cov)


    def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        """
        Chunk-based parametrization of TICA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance and time-lagged covariance
        matrices are estimated. Finally, the generalized eigenvalue problem is solved to determine
        the independent compoennts.

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
        logging.info("itraj = "+str(itraj)+". t = "+str(t)+". last_chunk_in_traj = "+str(last_chunk_in_traj)
                     +" last_chunk = "+str(last_chunk)+" ipass = "+str(ipass))

        if ipass == 0:
            # TODO: use a more advanced algo for mean calculation

            self.mu += np.sum(X, axis=0)
            self.N += np.shape(X)[0]

            if last_chunk:
                self.mu /= self.N
                print "mean:\n", self.mu

        if ipass == 1:

            assert Y is not None, "time lagged input missing"

            X_meanfree = X - self.mu
            Y_meanfree = Y - self.mu
            self.cov += np.dot(X_meanfree.T, X_meanfree)
            self.cov_tau += np.dot(X_meanfree.T, Y_meanfree)

            if last_chunk:
                self.cov /= self.N
                self.cov_tau /= self.N
                return True # finished!

        return False # not finished yet.


    def param_finish(self):
        """
        Finalizes the parametrization.

        :return:
        """
        if self.symmetrize:
            self.cov_tau = (self.cov_tau + self.cov_tau.T) / 2.

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
        self.lambdas = lambdas[abs_order_descend]
        V_tau = V_tau[:, abs_order_descend]

        # Compute U
        self.U = np.dot(
            W, np.dot(np.diag(1 / sigmaPC.squeeze()), V_tau))


    def map(self, X):
        """
        Projects the data onto the dominant independent components.

        :param X: the input data
        :return: the projected data
        """
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.U[:, 0:self.output_dim])
        return Y
