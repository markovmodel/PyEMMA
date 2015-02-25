__author__ = 'noe'

import numpy as np
from .transformer import Transformer
from pyemma.util.log import getLogger
from pyemma.util.annotators import doc_inherit

log = getLogger('PCA')
__all__ = ['PCA']


class PCA(Transformer):

    r"""Principal component analysis.

    Given a sequence of multivariate data :math:`X_t`,
    computes the mean-free covariance matrix.

    .. math:: C = (X - \mu)^T (X - \mu)

    and solves the eigenvalue problem

    .. math:: C r_i = \sigma_i r_i,

    where :math:`r_i` are the principal components and :math:`\sigma_i` are
    their respective variances.

    When used as a dimension reduction method, the input data is projected onto
    the dominant principal components.

    Parameters
    ----------
    output_dimension : int
        number of principal components to project onto

    """

    def __init__(self, output_dimension):
        super(PCA, self).__init__()
        self._output_dimension = output_dimension
        self._dot_prod_tmp = None
        self.Y = None

    @doc_inherit
    def describe(self):
        return "[PCA, output dimension = %i]" % self._output_dimension

    def dimension(self):
        """
        Returns the number of output dimensions

        :return:
        """
        return self._output_dimension

    @doc_inherit
    def get_constant_memory(self):
        """Returns the constant memory requirements, in bytes."""
        # memory for mu, C, v, R
        dim = self.data_producer.dimension()

        cov_elements = dim ** 2
        mu_elements = dim

        v_elements = dim
        R_elements = cov_elements

        return 8 * (cov_elements + mu_elements + v_elements + R_elements)

    @doc_inherit
    def get_memory_per_frame(self):
        # memory for temporaries
        dim = self.data_producer.dimension()

        x_meanfree_elements = self.chunksize * dim

        dot_prod_elements = dim

        return 8 * (x_meanfree_elements + dot_prod_elements)

    @doc_inherit
    def param_init(self):
        self.N = 0
        # create mean array and covariance matrix
        dim = self.data_producer.dimension()
        log.info("Running PCA on %i dimensional input" % dim)
        assert dim > 0, "Incoming data of PCA has 0 dimension!"
        self.mu = np.zeros(dim)
        self.C = np.zeros((dim, dim))

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj,
                       last_chunk, ipass, Y=None):
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
        # pass 1: means
        if ipass == 0:
            if t == 0:
                log.debug("start to calculate mean")
                self._sum_tmp = np.empty(X.shape[1])
            np.sum(X, axis=0, out=self._sum_tmp)
            self.mu += self._sum_tmp
            self.N += np.shape(X)[0]
            if last_chunk:
                self.mu /= self.N

        # pass 2: covariances
        if ipass == 1:
            if t == 0:
                log.debug("start calculate covariance")
                self._dot_prod_tmp = np.empty_like(self.C)
            Xm = X - self.mu
            np.dot(Xm.T, Xm, self._dot_prod_tmp)
            self.C += self._dot_prod_tmp
            if last_chunk:
                self.C /= self.N - 1
                log.debug("finished")
                return True  # finished!

        # by default, continue
        return False

    @doc_inherit
    def param_finish(self):
        (v, R) = np.linalg.eigh(self.C)
        # sort
        I = np.argsort(v)[::-1]
        self.v = v[I]
        self.R = R[:, I]

    def map(self, X):
        """
        Projects the data onto the dominant principal components.

        :param X: the input data
        :return: the projected data
        """
        X_meanfree = X - self.mu
        Y = np.dot(X_meanfree, self.R[:, 0:self._output_dimension])
        return Y
