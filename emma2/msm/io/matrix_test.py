"""Unit tests for matrix io implementations"""
import os
import unittest

import numpy as np
import scipy.sparse

import matrix

################################################################################
# ascii
################################################################################

################################################################################
# dense
################################################################################

class TestReadMatrixDense(unittest.TestCase):
    
    def setUp(self):
        self.filename_int='test/matrix_int.dat'
        self.filename_float='test/matrix_float.dat'
        self.filename_complex='test/matrix_complex.dat'        

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
        
class TestWriteMatrixDense(unittest.TestCase):

    def setUp(self):
        self.filename_int='test/matrix_int_out.dat'
        self.filename_float='test/matrix_float_out.dat'
        self.filename_complex='test/matrix_complex_out.dat'
        
        self.A_int=np.arange(3*3).reshape(3, 3)
        self.A_float=1.0*self.A_int
        self.A_complex=np.arange(3*3).reshape(3, 3)+\
            1j*np.arange(9,3*3+9).reshape(3, 3)

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)

    def test_write_matrix_dense(self):
        matrix.write_matrix_dense(self.filename_int, self.A_int, fmt='%d')
        An=np.loadtxt(self.filename_int, dtype=np.int)
        self.assertTrue(np.all(An==self.A_int))

        matrix.write_matrix_dense(self.filename_float, self.A_float)
        An=np.loadtxt(self.filename_int)
        self.assertTrue(np.all(An==self.A_float))

        matrix.write_matrix_dense(self.filename_complex, self.A_complex)
        An=np.loadtxt(self.filename_complex, dtype=np.complex)
        self.assertTrue(np.all(An==self.A_complex))

################################################################################
# sparse
################################################################################

class TestReadMatrixSparse(unittest.TestCase):       

    def setUp(self):
        self.filename_int='test/spmatrix_int.coo.dat'
        self.filename_float='test/spmatrix_float.coo.dat'
        self.filename_complex='test/spmatrix_complex.coo.dat'

        """Reference matrices in dense storage"""
        self.reference_int='test/spmatrix_int_reference.dat'
        self.reference_float='test/spmatrix_float_reference.dat'
        self.reference_complex='test/spmatrix_complex_reference.dat'

    def tearDown(self):
        pass

    def test_read_matrix_sparse(self):
        A=np.loadtxt(self.reference_int, dtype=np.int)
        A_n=matrix.read_matrix_sparse(self.filename_int, dtype=np.int).toarray()
        self.assertTrue(np.all(A==A_n))

        A=np.loadtxt(self.reference_float)
        A_n=matrix.read_matrix_sparse(self.filename_float).toarray()
        self.assertTrue(np.all(A==A_n))

        A=np.loadtxt(self.reference_complex, dtype=np.complex)
        A_n=matrix.read_matrix_sparse(self.filename_complex, dtype=np.complex).toarray()
        self.assertTrue(np.all(A==A_n))

class TestWriteMatrixSparse(unittest.TestCase):

    def is_integer(self, x):
        """Check if elements of an array can be represented by integers.
        
        Parameters 
        ----------
        x : ndarray
            Array to check.
        
        Returns
        -------
        is_int : ndarray of bool
            is_int[i] is True if x[i] can be represented
            as int otherwise is_int[i] is False.
        
        """
        is_int=np.equal(np.mod(x, 1), 0)
        return is_int

    def sparse_matrix_from_coo(self, coo):
        row=coo[:, 0]
        col=coo[:, 1]
        values=coo[:, 2]

        """Check if imaginary part of row and col is zero"""
        if np.all(np.isreal(row)) and np.all(np.isreal(col)):
            row=row.real
            col=col.real

            """Check if first and second column contain only integer entries"""
            if np.all(self.is_integer(row)) and np.all(self.is_integer(col)):           

                """Convert row and col to int"""
                row=row.astype(int)
                col=col.astype(int)

                """Create coo-matrix"""
                A=scipy.sparse.coo_matrix((values,(row, col)))
                return A
            else:
                raise ValueError('coo contains non-integer entries for row and col.')        
        else:
            raise ValueError('coo contains complex entries for row and col.')
        

    def extract_coo(self, A):
        A=A.tocoo()
        coo=np.transpose(np.vstack((A.row, A.col, A.data)))
        return coo

    def setUp(self):
        self.filename_int='test/spmatrix_int_out.coo.dat'
        self.filename_float='test/spmatrix_float_out.coo.dat'
        self.filename_complex='test/spmatrix_complex_out.coo.dat'

        """Tri-diagonal test matrices"""
        dim=10
        d0=np.arange(0, dim)
        d1=np.arange(dim, 2*dim-1)
        d_1=np.arange(2*dim, 3*dim-1)

        self.A_int=scipy.sparse.diags((d0, d1, d_1), (0, 1, -1), dtype=np.int).tocoo()
        self.A_float=scipy.sparse.diags((d0, d1, d_1), (0, 1, -1)).tocoo()
        self.A_complex=self.A_float+1j*self.A_float

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)
    
    def test_write_matrix_sparse(self):
        matrix.write_matrix_sparse(self.filename_int, self.A_int, fmt='%d')
        coo_n=np.loadtxt(self.filename_int, dtype=np.int)
        """Create sparse matrix from coo data"""
        A_n=self.sparse_matrix_from_coo(coo_n)
        diff=(self.A_int-A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data==0.0))        

        matrix.write_matrix_sparse(self.filename_float, self.A_float)
        coo_n=np.loadtxt(self.filename_int, dtype=np.float)
        """Create sparse matrix from coo data"""
        A_n=self.sparse_matrix_from_coo(coo_n)
        diff=(self.A_float-A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data==0.0))

        matrix.write_matrix_sparse(self.filename_complex, self.A_complex)
        coo_n=np.loadtxt(self.filename_complex, dtype=np.complex)
        """Create sparse matrix from coo data"""
        A_n=self.sparse_matrix_from_coo(coo_n)
        diff=(self.A_complex-A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data==0.0))

if __name__=="__main__":
    unittest.main()
        
