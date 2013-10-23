"""Implementation of IO for dense and sparse matrices"""

import numpy as np
import scipy.sparse

def is_integer(x):
    """Check if elements of array are integers
    
    Parameters 
    ----------
    x : ndarray
        Array to check.
    
    Returns
    -------
    is_int : ndarray of bool
        is_int[i] is True if x[i] is int otherwise False.
        
    """
    return np.equal(np.mod(x, 1), 0)
    
def read_matrix_dense(filename, dtype=float, comments='#'):
    A=np.loadtxt(filename, dtype=dtype, comments=comments)
    return A

def read_matrix_sparse(filename, dtype=float, comments='#'):
    coo=np.loadtxt(filename, comments=comments)
    
    """Check if coo is (M, 3) ndarray"""
    if len(coo.shape)==2 and coo.shape[1]==3:
        row=coo_data[:, 0]
        col=coo_data[:, 1]

        """Convert values to specified data-type"""
        values=coo_data[:, 2].astype(dtype)

        """Check if first and second column contain integer entries"""
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
        raise ValueError('Given file is not a sparse matrix in coo-format.')

