r"""This module implements the countmatrix estimation functionality

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse

################################################################################
# count_matrix
################################################################################

def count_matrix_mult(dtrajs, lag, sliding=True, sparse=True):
    r"""Generate a count matrix from a given list of discrete trajectories.    

    Parameters
    ----------
    dtrajs : list of array_like
        Discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix.
        
    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
       
    """
    nmax=0
    """Determine maximum microstate index over all trajectories"""
    if sliding:
        for dtraj in dtrajs:
            nmax=max(nmax, dtraj.max())
    else:
        for dtraj in dtrajs:
            nmax=max(nmax, max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max()))
    ndim=nmax+1        
    """If ndim<4000 use bincount else use coo"""
    if ndim<4000:
        return count_matrix_bincount_mult(dtrajs, lag, sliding=sliding, ndim=ndim, sparse=sparse)
    else:
        return count_matrix_coo_mult(dtrajs, lag, sliding=sliding, sparse=sparse)     

def count_matrix(dtraj, lag, sliding=True, sparse=True):
    r"""Generate a count matrix from a given list of integers.

    Parameters
    ----------
    dtraj : array_like
        Discretized trajectory
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format

    """
    if sliding:
        """Determine dimension of state space"""
        nmax=dtraj.max()
        ndim=nmax+1
    else:
        nmax=max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max())
        ndim=nmax+1

    """If ndim<4000 use bincount else use coo"""
    if ndim<4000:
        return count_matrix_bincount(dtraj, lag, sliding=sliding, ndim=ndim, sparse=sparse)
    else:
        return count_matrix_coo(dtraj, lag, sliding=sliding, sparse=sparse)
        
    

################################################################################
# coo
################################################################################

def count_matrix_coo(dtraj, lag, sliding=True, sparse=True):
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
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix.

    Returns
    -------
    C : scipy.sparse.csr_matrix
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

    C=C.tocsr()
    if C.shape[0] != C.shape[1]:
        C=make_square_coo_matrix(C)

    if sparse:
        return C.tocsr()
    else:
        return C.toarray()

def count_matrix_coo_mult(dtrajs, lag, sliding=True, sparse=True):
    r"""Generate a count matrix from a given list of discrete trajectories.

    The generated count matrix is a sparse matrix in coordinate 
    list (COO) format. 

    Parameters
    ----------
    dtrajs : list of array_like
        Discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
    
    """
    Z = scipy.sparse.coo_matrix((1,1))
    for dtraj in dtrajs:
        Zi = count_matrix_coo(dtraj, lag, sliding)
        Z = add_coo_matrix(Z, Zi)
    C=make_square_coo_matrix(Z)

    if sparse:
        return C.tocsr()
    else:
        return C.toarray()

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

################################################################################
# bincount
################################################################################

def count_matrix_bincount(dtraj, lag, sliding=True, sparse=True, ndim=None):
    r"""Generate a count matrix from a discrete trajectory.

    Parameters
    ----------
    dtraj : list of array_like
        Discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix.
    ndim : int (optional)
        The dimension of the count-matrix, ndim=nmax+1, where
        nmax is the maximum microstate index.

    Returns
    -------
    C : (M, M) ndarray or scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.

    Notes
    -----
    For Markov chains with less than 4000 states the use of 
    np.bincount and dense arrays seems to be faster than using
    scipy.sparse.coo_matrix to generate the count matrix.
    
    """
    dtraj=np.asarray(dtraj)
    M=len(dtraj)
    if(lag>M):
        raise ValueError("Value for lag is greater than "+\
                             "total length of given trajectory.")
    if sliding:
        """Determine dimension of state space"""
        if ndim is None:
            nmax=dtraj.max()
            ndim=nmax+1
        """Trajectory of flattend count-matrix indices k(i,j)=ndim*i+j"""
        ds=ndim*dtraj[0:-lag]+dtraj[lag:]        
    else:
        """Determine dimension of state space"""
        if ndim is None:
            nmax=max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max())
            ndim=nmax+1
        """Trajectory of flattend count-matrix indices k(i,j)=ndim*i+j"""
        ds=ndim*dtraj[0:-lag:lag]+dtraj[lag::lag]
    C=np.bincount(ds, minlength=ndim*ndim).reshape((ndim, ndim))
    if sparse:
        return scipy.sparse.csr_matrix(C)
    else:
        return C

def count_matrix_bincount_mult(dtrajs, lag, sliding=True, sparse=True, ndim=None):
    r"""Generate a count matrix from a given list of discrete trajectories.
    
    Parameters
    ----------
    dtrajs : list of array_like
        Discretized trajectories
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.
    ndim : int (optional)
        The dimension of the count-matrix, ndim=nmax+1, where
        nmax is the maximum microstate index.
        
    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
        
    """
    if ndim is None:
        nmax=0
        """Determine maximum microstate index over all trajectories"""
        if sliding:
            for dtraj in dtrajs:
                nmax=max(nmax, dtraj.max())
        else:
            for dtraj in dtrajs:
                nmax=max(nmax, max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max()))
        ndim=nmax+1        

    C=np.zeros((ndim, ndim))
    """Estimate count matrix for each discrete trajectory and add to C"""
    for dtraj in dtrajs:
        C+=count_matrix_bincount(dtraj, lag, sliding=sliding, sparse=False, ndim=ndim)
    if sparse:
        return scipy.sparse.csr_matrix(C)
    else:
        return C                   





