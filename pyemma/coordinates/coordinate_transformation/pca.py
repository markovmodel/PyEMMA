__author__ = 'noe'

import numpy as np
from transformer import Transformer


class PCA(Transformer):
    """
    Principal component analysis.

    Given a sequence of multivariate data X_t, computes the mean-free covariance matrix
    C = (X-mu)^T (X-mu)
    and solves the eigenvalue problem
    C r_i = sigma_i r_i,
    where r_i are the principal compontns and sigma_i are their respective variances.

    When used as a dimension reduction method, the input data is projected onto the
    dominant principal components.

    """

    def __init__(self, data_producer, output_dimension):
        """
        Constructs a feature reader

        :param trajectories:
            list of trajectory files

        :param structurefile:
            structure file (e.g. pdb)

        """
        self.data_producer = data_producer
        self.output_dimension = output_dimension

        # matrices
        self.N = 0
        self.mu = None
        self.C = None
        self.param_finished = False
        self.v = None
        self.R = None

    def describe(self):
        return "PCA, output dimension = ", self.output_dimension

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self.output_dimension

    def get_constant_memory(self):
        """
        Returns the constant memory requirements, in bytes

        :return:
        """
        # memory for mu, C, v, R
        return 2 * self.data_producer.dimension() * (self.data_producer.dimension() + 1)

    def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        """
        Chunk-based parametrization of PCA. Iterates through all data twice. In the first pass, the
        data means are estimated, in the second pass the covariance matrix is estimated.
        Finally, the eigenvalue problem is solved to determine the principal compoennts.

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
        print "itraj = ", itraj, "t = ", t, "last_chunk_in_traj = ", last_chunk_in_traj, "last_chunk = ", last_chunk, "ipass = ", ipass
        # pass 1
        if ipass == 0:
            if first_chunk:
                self.mu = np.zeros((self.data_producer.dimension()))
            self.mu += np.sum(X, axis=0)
            self.N += np.shape(X)[0]
            if last_chunk:
                self.mu /= self.N
        # pass 2
        if ipass == 1:
            if first_chunk:
                self.C = np.zeros(
                    (self.data_producer.dimension(), self.data_producer.dimension()))
            Xm = X - self.mu
            self.C += np.dot(Xm.T, Xm)
            if last_chunk:
                self.C /= self.N
                # diagonalize
                (v, R) = np.linalg.eig(self.C)
                # sort
                I = np.argsort(v)[::-1]
                self.v = v[I]
                self.R = R[I, :]
                # parametrization finished
                self.param_finished = True
                print "parametrization finished!"

    def map(self, X):
        """
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        Y = np.dot(X, self.R[:, 0:self.output_dimension])
        return Y
