"""This module implements the countmatrix estimation functionality"""

import numpy as np
import scipy.sparse

def count_matrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix from a given list of integers.

    The generated count matrix is a sparse matrix in coordinate 
    list (COO) format. 

    Parameters
    ----------
    dtraj : array_like
        Discretized trajectory
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The countmatrix at given lag in coordinate list format.
    
    """
    dtraj=np.asarray(dtraj)
    M=len(dtraj)
    if(lag>M):
        raise ValueError("Value for lag is greater than "+\
                             "total length of given trajectory.")

    if(sliding):
        row=dtraj[0:-lag]
        col=dtraj[lag:]
        N=row.shape[0]
        data=np.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)))

    else:
        row=dtraj[0:-lag:lag]
        col=dtraj[lag::lag]
        N=row.shape[0]
        data=np.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)))        

    C=C.tocsr().tocoo()
    C=make_square_coo_matrix(C)
    return C


def make_square_coo_matrix(A):
    r"""Reshape a COO sparse matrix to a square matrix.

    Transform a given sparse matrix in coordinate list (COO) format 
    of shape=(m, n) into a square matrix of shape=(N, N) with 
    N=max(m, n). The transformed matrix is also stored in coordinate
    list (COO) format.
    
    Parameters
    ----------
    A : scipy.sparse.coo_matrix
        Sparse matrix in coordinate list format
    
    Returns
    -------
    A_sq : scipy.sparse.coo_matrix
        Square sparse matrix in coordinate list format.
    
    """
    A=A.tocoo()
    N=max(A.shape)
    A_sq=scipy.sparse.coo_matrix((A.data, (A.row, A.col)), shape=(N, N))
    return A_sq    
    
def add_coo_matrix(A, B):
    """
    Add two sparse matrices in coordinate list (COO) format 
    with possibly incosistent shapes. If A is (k,l) shaped and
    B has shape (m, n) than C=A+B has shape (max(k, m), max(l, n)).

    Parameters :
    ------------
    A : scipy.sparse.coo_matrix
        Sparse matrix in coordinate list format
    B : scipy.sparse.coo_matrix
        Sparse matrix in coordinate list format

    Returns :
    ---------
    C : scipy.sparse.coo_matrix
        Sparse matrix in coordinate list format 

    """
    A=A.tocoo()
    B=B.tocoo()
    data=np.hstack((A.data, B.data))
    row=np.hstack((A.row, B.row))
    col=np.hstack((A.col, B.col))
    C=scipy.sparse.coo_matrix((data, (row, col)))
    return C.tocsr().tocoo()

