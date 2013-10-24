"""Implementation of IO for dense and sparse matrices"""

import numpy as np
import scipy.sparse

################################################################################
# ascii
################################################################################

################################################################################
# dense
################################################################################
    
def read_matrix_dense(filename, dtype=float, comments='#'):
    A=np.loadtxt(filename, dtype=dtype, comments=comments)
    return A

def write_matrix_dense(filename, A, fmt='%.18e', header='', comments='#'):
    np.savetxt(filename, A, fmt=fmt, header=header, comments=comments)

################################################################################
# sparse
################################################################################

def is_integer(x):
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

def read_matrix_sparse(filename, dtype=float, comments='#'):
    coo=np.loadtxt(filename, comments=comments, dtype=dtype)
    
    """Check if coo is (M, 3) ndarray"""
    if len(coo.shape)==2 and coo.shape[1]==3:
        row=coo[:, 0]
        col=coo[:, 1]
        values=coo[:, 2]

        """Check if imaginary part of row and col is zero"""
        if np.all(np.isreal(row)) and np.all(np.isreal(col)):
            row=row.real
            col=col.real

            """Check if first and second column contain only integer entries"""
            if np.all(is_integer(row)) and np.all(is_integer(col)):           

                """Convert row and col to int"""
                row=row.astype(int)
                col=col.astype(int)

                """Create coo-matrix"""
                A=scipy.sparse.coo_matrix((values,(row, col)))
                return A
            else:
                raise ValueError('File contains non-integer entries for row and col.')        
        else:
            raise ValueError('File contains complex entries for row and col.')
    else:
        raise ValueError('Given file is not a sparse matrix in coo-format.')

def write_matrix_sparse(filename, A, fmt='%.18e', header='', comments='#'):
    if scipy.sparse.issparse(A):
        """Convert to coordinate list (coo) format"""
        A=A.tocoo()
        coo=np.transpose(np.vstack((A.row, A.col, A.data)))
        np.savetxt(filename, coo, fmt=fmt, header=header, comments=comments)
    else:
        raise ValueError('The given matrix is not a sparse matrix')

################################################################################
# binary
################################################################################

################################################################################
# dense
################################################################################

def load_matrix_dense(filename):
    A=np.load(filename)
    return A    

def save_matrix_dense(filename, A): 
    np.save(filename, A)

################################################################################
# sparse
################################################################################

def load_matrix_sparse(filename):
    coo=np.load(filename)
    
    """Check if coo is (M, 3) ndarray"""
    if len(coo.shape)==2 and coo.shape[1]==3:
        row=coo[:, 0]
        col=coo[:, 1]
        values=coo[:, 2]

        """Check if imaginary part of row and col is zero"""
        if np.all(np.isreal(row)) and np.all(np.isreal(col)):
            row=row.real
            col=col.real

            """Check if first and second column contain only integer entries"""
            if np.all(is_integer(row)) and np.all(is_integer(col)):           

                """Convert row and col to int"""
                row=row.astype(int)
                col=col.astype(int)

                """Create coo-matrix"""
                A=scipy.sparse.coo_matrix((values,(row, col)))
                return A
            else:
                raise ValueError('File contains non-integer entries for row and col.')        
        else:
            raise ValueError('File contains complex entries for row and col.')
    else:
        raise ValueError('Given file is not a sparse matrix in coo-format.')

def save_matrix_sparse(filename, A):
    if scipy.sparse.issparse(A):
        """Convert to coordinate list (coo) format"""
        A=A.tocoo()
        coo=np.transpose(np.vstack((A.row, A.col, A.data)))
        np.save(filename, coo)
    else:
        raise ValueError('The given matrix is not a sparse matrix')
