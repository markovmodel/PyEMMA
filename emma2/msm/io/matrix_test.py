"""Unit tests for matrix io implementations"""

import unittest

import numpy as np
import scipy.sparse

import matrix

class TestReadMatrixDense(unittest.TestCase):
    
    def setUp(self):
        self.filename_int='test/matrix_dense_int.dat'
        self.filename_float='test/matrix_dense_float.dat'
        self.filename_complex='test/matrix_dense_complex.dat'
        
        # self.A_int=np.arange(3*3).reshape(3, 3)
        # self.A_float=1.0*self.A_int
        # self.A_complex=np.arange(3*3).reshape(3, 3)+\
        #     1j*np.arange(9,3*3+9).reshape(3, 3)

        self.A_int=np.loadtxt(self.filename_int, dtype=np.int)
        self.A_float=np.loadtxt(self.filename_float, dtype=np.float)
        self.A_complex=np.loadtxt(self.filename_complex, dtype=np.complex)

    def tearDown(self):
        pass

    def test_read_matrix_dense(self):
        A=matrix.read_matrix_dense(self.filename_int, dtype=np.int)
        self.assertTrue(np.all(A==self.A_int))

        A=matrix.read_matrix_dense(self.filename_float)
        self.assertTrue(np.all(A==self.A_float))

        A=matrix.read_matrix_dense(self.filename_complex, dtype=np.complex)
        self.assertTrue(np.all(A==self.A_complex))
        

if __name__=="__main__":
    unittest.main()
        
