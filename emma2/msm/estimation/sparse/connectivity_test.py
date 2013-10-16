"""Unit tests for the connectivity module"""

import unittest

import numpy as np
import scipy.sparse

import connectivity

class TestConneectedSets(unittest.TestCase):
    
    def setUp(self):
        self.nc=3
        C1=np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2=np.array([[0, 1], [1, 0]])
        C3=np.array([[7]])

        self.C=scipy.sparse.block_diag((C1, C2, C3))
        self.cc=[np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        

    def tearDown(self):
        pass

    def test_connected_sets(self):
        cc=connectivity.connected_sets(self.C)
        for i in range(self.nc):
            self.assertTrue(np.all(self.cc[i]==np.sort(cc[i])))

if __name__=="__main__":
    unittest.main()
