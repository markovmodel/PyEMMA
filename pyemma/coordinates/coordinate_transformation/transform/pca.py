__author__ = 'noe'

import numpy as np
from pyemma.coordinates.coordinate_transformation.transform.transformer import Transformer
from pyemma.util.log import getLogger

log = getLogger('PCA')
__all__ = ['PCA']


class PCA(Transformer):
    r"""Principal component analysis.

    Given a sequence of multivariate data X_t, computes the mean-free covariance matrix
    C = (X-mu)^T (X-mu)
    and solves the eigenvalue problem
    C r_i = sigma_i r_i,
    where r_i are the principal components and sigma_i are their respective variances.

    When used as a dimension reduction method, the input data is projected onto the
    dominant principal components.

    """

    def __init__(self, output_dimension):
        """
        Constructs a PCA transformation

        :param output_dimension:
            number of principal components to project onto

        :param structurefile:
            structure file (e.g. pdb)

        """
        super(PCA, self).__init__(self)
        self.output_dimension = output_dimension

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


    def param_init(self):
        """
        Initializes the parametrization.

        :return:
        """
        log.info("Running PCA")
        self.N = 0
        # create mean array and covariance matrix
        self.mu = np.zeros((self.data_producer.dimension()))
        self.C = np.zeros((self.data_producer.dimension(), self.data_producer.dimension()))


    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
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
        log.info("itraj = "+str(itraj)+". t = "+str(t)+". last_chunk_in_traj = "+str(last_chunk_in_traj)
                     +" last_chunk = "+str(last_chunk)+" ipass = "+str(ipass))

        # pass 1: means
        if ipass == 0:
            self.mu += np.sum(X, axis=0)
            self.N += np.shape(X)[0]
            if last_chunk:
                self.mu /= self.N

        # pass 2: covariances
        if ipass == 1:
            Xm = X - self.mu
            self.C += np.dot(Xm.T, Xm)
            if last_chunk:
                self.C /= self.N
                return True # finished!

        # by default, continue
        return False


    def param_finish(self):
        """
        Finalizes the parametrization.

        :return:
        """
        # diagonalize
        (v, R) = np.linalg.eig(self.C)
        # sort
        I = np.argsort(v)[::-1]
        self.v = v[I]
        self.R = R[I, :]


    def map(self, X):
        """
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        Y = np.dot(X, self.R[:, 0:self.output_dimension])
        return Y
