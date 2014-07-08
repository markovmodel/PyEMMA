"""Unit tests for the count_matrix module"""

import unittest

import numpy as np
import scipy.sparse

import count_matrix

from os.path import abspath, join
from os import pardir

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'

class TestCountMatrixMult(unittest.TestCase):
    
    def setUp(self):
        """Small test cases"""
        self.S1=np.array([0, 0, 0, 1, 1, 1])
        self.S2=np.array([0, 0, 0, 1, 1, 1])

        self.B1_sliding=np.array([[4, 2], [0, 4]])
        self.B2_sliding=np.array([[2, 4], [0, 2]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C=count_matrix.count_matrix_mult([self.S1,self.S2], 1, sliding=True).toarray()
        self.assertTrue(np.allclose(C, self.B1_sliding))

        C=count_matrix.count_matrix_mult([self.S1,self.S2], 2, sliding=True).toarray()
        self.assertTrue(np.allclose(C, self.B2_sliding))


class TestCountMatrix(unittest.TestCase):
    
    def setUp(self):
        """Small test cases"""
        self.S_short=np.array([0, 0, 1, 0, 1, 1, 0])
        self.B1_lag=np.array([[1, 2], [2, 1]])
        self.B2_lag=np.array([[0, 1], [1, 1]])
        self.B3_lag=np.array([[2]])

        self.B1_sliding=np.array([[1, 2], [2, 1]])
        self.B2_sliding=np.array([[1, 2], [1, 1]])
        self.B3_sliding=np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        self.S_long=np.loadtxt(testpath + 'dtraj.dat')
        self.C1_lag=np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag=np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag=np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding=np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding=np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding=np.loadtxt(testpath + 'C_13_sliding.dat')


    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C=count_matrix.count_matrix(self.S_short, 1, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.B1_lag))

        C=count_matrix.count_matrix(self.S_short, 2, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.B2_lag))

        C=count_matrix.count_matrix(self.S_short, 3, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.B3_lag))

        C=count_matrix.count_matrix(self.S_short, 1).toarray()
        self.assertTrue(np.allclose(C, self.B1_sliding))

        C=count_matrix.count_matrix(self.S_short, 2).toarray()
        self.assertTrue(np.allclose(C, self.B2_sliding))

        C=count_matrix.count_matrix(self.S_short, 3).toarray()
        self.assertTrue(np.allclose(C, self.B3_sliding))

        """Larger test cases"""
        C=count_matrix.count_matrix(self.S_long, 1, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.C1_lag))

        C=count_matrix.count_matrix(self.S_long, 7, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.C7_lag))

        C=count_matrix.count_matrix(self.S_long, 13, sliding=False).toarray()
        self.assertTrue(np.allclose(C, self.C13_lag))

        C=count_matrix.count_matrix(self.S_long, 1).toarray()
        self.assertTrue(np.allclose(C, self.C1_sliding))

        C=count_matrix.count_matrix(self.S_long, 7).toarray()
        self.assertTrue(np.allclose(C, self.C7_sliding))

        C=count_matrix.count_matrix(self.S_long, 13).toarray()
        self.assertTrue(np.allclose(C, self.C13_sliding))

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C=count_matrix.count_matrix(self.S_short, 10)


class TestAddCooMatrix(unittest.TestCase):

    def setUp(self):
        An=np.array([[1.0, 2.0], [3.0, 4.0]])
        Bn=np.array([[-1.0, 3.0], [-4.0, 2.0]])
        Cn=np.array([[0.0, 5.0], [-1.0, 6.0]])

        self.A=scipy.sparse.coo_matrix(An)
        self.B=scipy.sparse.coo_matrix(Bn)
        self.C=scipy.sparse.coo_matrix(Cn)

    def tearDown(self):
        pass

    def test_add_coo_matrix(self):
        C_test=count_matrix.add_coo_matrix(self.A, self.B)
        self.assertTrue(np.allclose(C_test.toarray(), self.C.toarray()))


class TestMakeSquareCooMatrix(unittest.TestCase):
    
    def setUp(self):
        An=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.An_square=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
        self.A=scipy.sparse.coo_matrix(An)

        Bn=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.Bn_square=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
        self.B=scipy.sparse.coo_matrix(Bn)
        
    def tearDown(self):
        pass

    def test_make_square_coo_matrix(self):
        A_test=count_matrix.make_square_coo_matrix(self.A)
        self.assertTrue(np.allclose(A_test.toarray(), self.An_square))
        
        B_test=count_matrix.make_square_coo_matrix(self.B)
        self.assertTrue(np.allclose(B_test.toarray(), self.Bn_square))      
        
    
if __name__=="__main__":
    unittest.main()

