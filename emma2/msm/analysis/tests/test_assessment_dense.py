'''
Created on 07.10.2013

@author: marscher
'''
import unittest
import numpy as np

from birth_death_chain import BirthDeathChain

from emma2.msm.analysis import is_rate_matrix, is_reversible, is_transition_matrix, is_connected

def create_rate_matrix():
    a = [[-3, 3, 0, 0 ],
         [3, -5, 2, 0 ],
         [0, 3, -5, 2],
         [0, 0, 3, -3] ]
    
    return np.asmatrix(a)

class RateMatrixTest(unittest.TestCase):

    def setUp(self):
        self.A = create_rate_matrix()

    def test_IsRateMatrix(self):
        self.assert_(is_rate_matrix(self.A), \
                     'A should be a rate matrix')
        
        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(is_rate_matrix(self.A), \
                        'matrix is not a rate matrix')
        
class ReversibleTest(unittest.TestCase):
    
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()
        self.mu = self.bdc.stationary_distribution()
    
    def test_IsReversible(self):
        # create a reversible matrix
        self.assertTrue(is_reversible(self.T, self.mu),
                        "T should be reversible")

    def test_is_transition_matrix(self):
        self.assertTrue(is_transition_matrix(self.T))

    def test_is_connected(self):
        self.assertTrue(is_connected(self.T))
        self.assertTrue(is_connected(self.T, directed=False))

    

if __name__ == "__main__":
    unittest.main()
