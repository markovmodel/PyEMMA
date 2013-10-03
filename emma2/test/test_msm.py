'''
Created on 24.09.2013

.. moduleauthor: marscher <m.scherer@fu-berlin.de>

'''
import unittest
import numpy as np
from scipy.sparse import coo_matrix
from msm.analysis.api import *

class _TestMSM(unittest.TestCase):

    def test_is_transition_matrix_dense(self):
        """
            tests the function msm.util.isTransitionMatrix(T)
        """
        size = 100
        # create count matrix with samples between [0, 100] and shape size^2
        # and cast it to double precision
        C = np.asarray(np.random.randint(0, 100, size=(size, size)), np.float64, 2)
        T = C / C.sum(axis=1)[ : , np.newaxis]
        
        self.assert_(is_transition_matrix(T, 1e-15), "T should be a transition matrix")

    def test_is_transition_matrix_sparse(self):
        """
            tests the function msm.util.isTransitionMatrix(T)
        """
        size = 100
        # create count matrix with samples between [0, 100] and shape size^2
        # and cast it to double precision
        C = np.asarray(np.random.randint(0, 100, size=(size, size)), np.float64, 2)
        T = C / C.sum(axis=1)[ : , np.newaxis]
        
        self.assert_(is_transition_matrix(T, 1e-15), "T should be a transition matrix")


    def test_eigenvalues(self):
        result = np.asarray([[2,0,0], [0,3,4], [0,4,9]])
        exp_result = np.asarray([11.0, 2.0, 1.0])
        
        self.assert_(np.allclose(result, exp_result))
        print eigenvalues(result, 2)
        print eigenvalues(result, 1)
        print eigenvalues(result, (0,1))
        
    def test_is_rate_matrix(self):
        l = 10
        mu = 3
        
        rows = 3
        cols = 4
        
        data = []
        
        for r in xrange(0, rows):
            data.append(-l)
            data.append(l)
        
        Q = np.matrix(data)
        self.skipTest("not impled")
        self.fail("not yet impled")
        self.assert_(is_rate_matrix(Q), "Q should be a rate matrix")

    def test_eigenvectors(self):
        
        

if __name__ == "__main__":
    unittest.main()
