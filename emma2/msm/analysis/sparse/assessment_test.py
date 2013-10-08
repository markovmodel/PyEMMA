"""This module provides unit tests for the assessment module"""

import scipy.sparse
from scipy.sparse.construct import spdiags
from scipy.sparse.dia import dia_matrix
import unittest

import assessment
import numpy as np


def normalize_rows(A):
    """Normalize rows of sparse marix"""
    A=A.tocsr()
    values=A.data
    indptr=A.indptr #Row index pointers
    indices=A.indices #Column indices

    dim=A.shape[0]
    normed_values=np.zeros(len(values))
    
    for i in range(dim):
        thisrow=values[indptr[i]:indptr[i+1]]
        rowsum=np.sum(thisrow)
        normed_values[indptr[i]:indptr[i+1]]=thisrow/rowsum
    
    return scipy.sparse.csr_matrix((normed_values,indices,indptr))

def random_nonempty_rows(M, N, density=0.01):
    """Generate a random sparse matrix with nonempty rows"""
    N_el=int(density*M*N) # total number of non-zero elements
    if N_el<M:
        raise ValueError("Density too small to obtain nonempty rows")
    else :
        rows=np.zeros(N_el)
        rows[0:M]=np.arange(M)
        rows[M:N_el]=np.random.random_integers(0, M-1, size=(N_el-M,))
        cols=np.random.random_integers(0, N-1, size=(N_el,))
        values=np.random.rand(N_el)
        return scipy.sparse.coo_matrix((values, (rows, cols)))

class TestTransitionMatrix(unittest.TestCase):

    def setUp(self):
        self.dim=10000
        self.density=0.001
        self.tol=1e-15
        A=random_nonempty_rows(self.dim, self.dim, density=self.density)
        self.T=normalize_rows(A)

    def tearDown(self):
        pass

    def test_is_transition_matrix(self):
        self.assertTrue(assessment.is_transition_matrix(self.T, tol=self.tol))


class TestRateMatrix(unittest.TestCase):
    
    def create_sparse_rate_matrix(self):
        """
        constructs the following rate matrix for a M/M/1 queue
        TODO: fix math string
        :math: `
        Q = \begin{pmatrix}
        -\lambda & \lambda \\
        \mu & -(\mu+\lambda) & \lambda \\
        &\mu & -(\mu+\lambda) & \lambda \\
        &&\mu & -(\mu+\lambda) & \lambda &\\
        &&&&\ddots
        \end{pmatrix}`
        taken from: https://en.wikipedia.org/wiki/Transition_rate_matrix
        """
        lambda_ = 5
        mu = 3
        dim = self.dim
        
        diag = np.empty((3, dim))
        # main diagonal
        diag[0, 0] = (-lambda_)
        diag[0, 1:dim - 1] = -(mu + lambda_)
        diag[0, dim-1] = lambda_
        
        # lower diag
        diag[1, : ] = mu
        diag[1, -2 : ] = -mu
        # upper diag
        diag[2, : ] = lambda_
        
        offsets = [0, -1, 1]
        
        return spdiags(diag, offsets, dim, dim, format='csr')
        A = dia_matrix((diag, offsets), shape=(dim, dim))
        return A.tocsr()

    def setUp(self):
        self.dim = 1e6
        self.K = self.create_sparse_rate_matrix()
        self.tol = 1e-15
    
    def test_is_rate_matrix(self):
        K_copy = self.K.copy()
        self.assertTrue(assessment.is_rate_matrix(self.K, self.tol), \
                        "K should be evaluated as rate matrix.")
        self.assertTrue(np.allclose(self.K.data, K_copy.data) and \
                        np.allclose(self.K.indptr, K_copy.indptr) and \
                        np.allclose(self.K.indices, K_copy.indices), \
                        "object modified!")


if __name__=="__main__":
    unittest.main()
