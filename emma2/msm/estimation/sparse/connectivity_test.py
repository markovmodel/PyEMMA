"""Unit tests for the connectivity module"""

import unittest

import numpy as np
import scipy.sparse

import connectivity

class TestConnectedSets(unittest.TestCase):
    
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

class TestLargestConnectedSet(unittest.TestCase):

    def setUp(self):
        self.nc=3
        C1=np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2=np.array([[0, 1], [1, 0]])
        C3=np.array([[7]])

        self.C=scipy.sparse.block_diag((C1, C2, C3))
        self.cc=[np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]        
        self.lcc=self.cc[0]

    def tearDown(self):
        pass

    def test_largest_connected_set(self):
        lcc=connectivity.largest_connected_set(self.C)
        self.assertTrue(np.all(self.lcc==np.sort(lcc)))

class TestConnectedCountMatrix(unittest.TestCase):
        
    def setUp(self):
        self.nc=3
        C1=np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2=np.array([[0, 1], [1, 0]])
        C3=np.array([[7]])

        self.C=scipy.sparse.block_diag((C1, C2, C3))
        self.C_cc=C1
        self.cc=[np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]        
        self.lcc=self.cc[0]

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        C_cc=connectivity.connected_count_matrix(self.C)
        self.assertTrue(np.allclose(C_cc.toarray(), self.C_cc))       
        
class TestIsConnected(unittest.TestCase):
        
    def setUp(self):
        self.nc=3
        C1=np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2=np.array([[0, 1], [1, 0]])
        C3=np.array([[7]])
        
        self.CConnected = scipy.sparse.csr_matrix(C1);
        
        self.CNotConnected=scipy.sparse.block_diag((C1, C2, C3))

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        C_nc=connectivity.is_connected(self.CNotConnected)
        self.assertFalse(C_nc)     
        C_cc=connectivity.is_connected(self.CConnected)
        self.assertTrue(C_cc)     
        

if __name__=="__main__":
    unittest.main()
