'''
Created on 24.09.2013

@author: marscher
'''
import unittest
import numpy as np
from emma2.msm.estimate import *

class TestMSM(unittest.TestCase):

    def test_is_transition_matrix(self):
        """
            tests the function msm.util.isTransitionMatrix(T)
        """
        size = 10
        # create count matrix with samples between [0, 100] and shape size^2
        # and cast it to float
        C = np.asarray(np.random.randint(0, 100, size=(size, size)), np.float, 2)
        row_sums = C.sum(axis = 1)
        for i in xrange(size):
            C[i] /= row_sums[i]
        
        self.assert_(is_transitionmatrix(C), "T should be a transition matrix")


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

if __name__ == "__main__":
    unittest.main()