"""This module implements the countmatrix estimation functionality"""

import numpy
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
    M=len(dtraj)
    if(lag>M):
        raise ValueError("Value for lag is greater than "+\
                             "total length of given trajectory.")

    if(sliding):
        row=dtraj[0:-lag]
        col=dtraj[lag:]
        N=row.shape[0]
        data=numpy.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)))

    else:
        row=dtraj[0:-lag:lag]
        col=dtraj[lag::lag]
        N=row.shape[0]
        data=numpy.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)))        

    C=C.tocsr().tocoo()
    C=_make_square_coo_matrix(C)
    return C


def _make_square_coo_matrix(A):
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
    N=max(A.shape)
    A_sq=scipy.sparse.coo_matrix((A.data, (A.row, A.col)), shape=(N, N))
    return A_sq    
    
