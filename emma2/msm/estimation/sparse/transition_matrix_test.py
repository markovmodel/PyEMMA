import unittest

import numpy as np
import scipy.sparse

import transition_matrix

"""Unit tests for the transition_matrix module"""

class TestTransitionMatrixNonReversible(unittest.TestCase):
    
    def setUp(self):
        """Small test cases"""
        self.C1=scipy.sparse.csr_matrix([[1, 3], [3, 1]])
        self.C2=scipy.sparse.csr_matrix([[0, 2], [1, 1]])
        
        self.T1=scipy.sparse.csr_matrix([[0.25, 0.75], [0.75, 0.25]])
        self.T2=scipy.sparse.csr_matrix([[0, 1], [0.5, 0.5]])
        
        self.Cov2 = [[[ 0.0375, -0.0375],  [-0.0375,  0.0375]], [[ 0.05,  -0.05  ],  [-0.05,    0.05  ]]]
        
        """Zero row sum throws an error"""
        self.C0=scipy.sparse.csr_matrix([[0, 0], [3, 1]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        T=transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        self.assertTrue(np.allclose(T, self.T1.toarray()))
        
        T=transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        self.assertTrue(np.allclose(T, self.T1.toarray()))
        
    def test_tmatrix_cov(self):
        """Small test cases"""
        Cov = transition_matrix.tmatrix_cov(self.C2.toarray())
        self.assertTrue(np.allclose(Cov, self.Cov2))
        
if __name__=="__main__":
    unittest.main()
