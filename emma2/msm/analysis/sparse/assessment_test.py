"""This module provides unit tests for the assessment module"""

import unittest
import assessment

import numpy as np
import scipy
import scipy.sparse

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
   
class TestAssessment(unittest.TestCase):

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

if __name__=="__main__":
    unittest.main()
