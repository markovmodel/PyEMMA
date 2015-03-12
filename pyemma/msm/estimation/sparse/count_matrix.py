r"""This module implements the countmatrix estimation functionality

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse

################################################################################
# count_matrix
################################################################################

def count_matrix_mult(dtrajs, lag, sliding=True, sparse=True, nstates=None, failfast=False):
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
        Whether to return a dense or a sparse matrix
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates)
    failfast = False : bool (optional)
        True: will raise an error as soon as the lag time is longer than any of the trajectories.
        False: will perform as long as the lag time fits within at least one of the trajectories.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
       
    """

    """Determine maximum state index, nmax, over all trajectories"""
    nmax=0
    for dtraj in dtrajs:
        nmax=max(nmax, dtraj.max())

    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=nmax+1

    """Raise Error if nstates<nmax+1"""
    if nstates < nmax+1:
        raise ValueError("nstates is smaller than the number of observed microstates")

    # nmax=0
    # """Determine maximum microstate index over all trajectories"""
    # if sliding:
    #     for dtraj in dtrajs:
    #         nmax=max(nmax, dtraj.max())
    # else:
    #     for dtraj in dtrajs:
    #         nmax=max(nmax, max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max()))
    # nstates=nmax+1        

    """If nstates<4000 use bincount else use coo"""
    if nstates<4000:
        return count_matrix_bincount_mult(dtrajs, lag, sliding=sliding, nstates=nstates, sparse=sparse, failfast=failfast)
    else:
        return count_matrix_coo_mult(dtrajs, lag, sliding=sliding, sparse=sparse, failfast=failfast)

def count_matrix(dtraj, lag, sliding=True, sparse=True, nstates=None):
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
        Whether to return a dense or a sparse matrix
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates)

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format

    """
    # if sliding:
    #     """Determine dimension of state space"""
    #     nmax=dtraj.max()
    #     nstates=nmax+1
    # else:
    #     nmax=max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max())
    #     nstates=nmax+1

    """Dimension of state space is maximum microstate index + 1"""
    nmax=dtraj.max()
    
    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=nmax+1

    """Raise Error if nstates<nmax+1"""
    if nstates < nmax+1:
        raise ValueError("nstates is smaller than the number of observed microstates")

    """If nstates<4000 use bincount else use coo"""
    if nstates<4000:
        return count_matrix_bincount(dtraj, lag, sliding=sliding, sparse=sparse, nstates=nstates)
    else:
        return count_matrix_coo(dtraj, lag, sliding=sliding, sparse=sparse, nstates=nstates)
        
    

################################################################################
# coo
################################################################################

def count_matrix_coo(dtraj, lag, sliding=True, sparse=True, nstates=None):
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
        is used for transition counting
    sparse : bool (optional)
        Whether to return a dense or a sparse matrix
    nstates : int, optional
        Enforce a count-matrix with shape=(nstates, nstates)

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

    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=dtraj.max()+1

    if(sliding):
        row=dtraj[0:-lag]
        col=dtraj[lag:]
        N=row.shape[0]
        data=np.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)), shape=(nstates, nstates))        
    else:
        row=dtraj[0:-lag:lag]
        col=dtraj[lag::lag]
        N=row.shape[0]
        data=np.ones(N)
        C=scipy.sparse.coo_matrix((data, (row, col)), shape=(nstates, nstates))

    C=C.tocsr()
    if C.shape[0] != C.shape[1]:
        C=make_square_coo_matrix(C)

    if sparse:
        return C.tocsr()
    else:
        return C.toarray()

def count_matrix_coo_mult(dtrajs, lag, sliding=True, sparse=True, nstates=None, failfast=False):
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
    failfast = False : bool (optional)
        True: will raise an error as soon as the lag time is longer than any of the trajectories.
        False: will perform as long as the lag time fits within at least one of the trajectories.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
    
    """

    """Determine maximum state index, nmax, over all trajectories"""
    nmax=0
    for dtraj in dtrajs:
        nmax=max(nmax, dtraj.max())

    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=nmax+1

    C=scipy.sparse.coo_matrix((nstates, nstates))

    counted = False
    for dtraj in dtrajs:
        if lag < np.size(dtraj):
            Zi=count_matrix_coo(dtraj, lag, sliding=sliding, nstates=nstates)
            C=C+Zi
            counted = True
        elif failfast:
            raise ValueError('Lag time '+str(lag)+'is longer or equal than at least one trajectory. Either reduce lag time or set failfast=False')
        else:
            pass # nothing to do

    if not counted:
        raise ValueError('Lag time '+str(lag)+'is longer or equal than all trajectories. Reduce lag time.')

    # Z = scipy.sparse.coo_matrix((1,1))
    # for dtraj in dtrajs:
    #     Zi = count_matrix_coo(dtraj, lag, sliding)
    #     Z = add_coo_matrix(Z, Zi)
    # C=make_square_coo_matrix(Z)

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

def count_matrix_bincount(dtraj, lag, sliding=True, sparse=True, nstates=None):
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
    nstates : int (optional)
        The dimension of the count-matrix, nstates=nmax+1, where
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

    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=dtraj.max()+1

    if sliding:
        # """Determine dimension of state space"""
        # if nstates is None:
        #     nmax=dtraj.max()
        #     nstates=nmax+1
        """Trajectory of flattend count-matrix indices k(i,j)=nstates*i+j"""
        ds=nstates*dtraj[0:-lag]+dtraj[lag:]        
    else:
        # """Determine dimension of state space"""
        # if nstates is None:
        #     nmax=max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max())
        #     nstates=nmax+1
        """Trajectory of flattend count-matrix indices k(i,j)=nstates*i+j"""
        ds=nstates*dtraj[0:-lag:lag]+dtraj[lag::lag]
    C=np.bincount(ds, minlength=nstates*nstates).reshape((nstates, nstates))
    if sparse:
        return scipy.sparse.csr_matrix(C)
    else:
        return C

def count_matrix_bincount_mult(dtrajs, lag, sliding=True, sparse=True, nstates=None, failfast=False):
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
    nstates : int (optional)
        The dimension of the count-matrix, nstates=nmax+1, where
        nmax is the maximum microstate index.
    failfast = False : bool (optional)
        True: will raise an error as soon as the lag time is longer than any of the trajectories.
        False: will perform as long as the lag time fits within at least one of the trajectories.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        The countmatrix at given lag in coordinate list format.
        
    """

    """Determine maximum state index, nmax, over all trajectories"""
    nmax=0
    for dtraj in dtrajs:
        nmax=max(nmax, dtraj.max())

    """Default is nstates = number of observed states at lagtime=1"""
    if nstates is None:
        nstates=nmax+1

    # if nstates is None:
    #     nmax=0
    #     """Determine maximum microstate index over all trajectories"""
    #     if sliding:
    #         for dtraj in dtrajs:
    #             nmax=max(nmax, dtraj.max())
    #     else:
    #         for dtraj in dtrajs:
    #             nmax=max(nmax, max(dtraj[0:-lag:lag].max(), dtraj[lag::lag].max()))
    #     nstates=nmax+1        

    C=np.zeros((nstates, nstates))
    """Estimate count matrix for each discrete trajectory and add to C"""

    counted = False
    for dtraj in dtrajs:
        if lag < np.size(dtraj):
            C+=count_matrix_bincount(dtraj, lag, sliding=sliding, sparse=False, nstates=nstates)
            counted = True
        elif failfast:
            raise ValueError('Lag time '+str(lag)+'is longer or equal than at least one trajectory. Either reduce lag time or set failfast=False')
        else:
            pass # nothing to do

    if not counted:
        raise ValueError('Lag time '+str(lag)+'is longer or equal than all trajectories. Reduce lag time.')


    if sparse:
        return scipy.sparse.csr_matrix(C)
    else:
        return C                   





