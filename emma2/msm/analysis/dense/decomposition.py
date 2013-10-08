"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the dense implementations for functions specified in msm.api. 
Dense matrices are represented by numpy.ndarrays throughout this module.
"""

import numpy as np
from scipy.linalg import eig, eigvals

import assessment

def mu(T):
    r"""Compute stationary distribution of stochastic matrix T. 
      
    The stationary distribution is the left eigenvector corresponding to the 
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns:
    --------
    mu : numpy array, shape(d,)      
        Vector of stationary probabilities.

    """
    val, L  = eig(T, left=True, right=False)

    """ Sorted eigenvalues and left and right eigenvectors. """
    perm=np.argsort(val)[::-1]

    val=val[perm]
    L=L[:,perm]
    """ Make sure that stationary distribution is non-negative and l1-normalized """
    nu=np.abs(L[:,0])
    mu=nu/np.sum(nu)
    return mu
    
def eigenvalues(T, k=None):
    r"""Compute eigenvalues of given transition matrix.
    
    Eigenvalues are computed using the numpy.linalg interface 
    for the corresponding LAPACK routines.    

    Input
    -----
    T : numpy.ndarray, shape=(d,d)
        Transition matrix (stochastic matrix).
    k : int (optional) or tuple of ints
        Compute the first k eigenvalues of T.

    Returns
    -------
    eig : numpy.ndarray, shape(n,)
        The eigenvalues of T ordered with decreasing absolute value.
        If k is None then n=d, if k is int then n=k otherwise
        n is the length of the given tuple of eigenvalue indices.

    """
    evals = np.sort(eigvals(T))[::-1]
    if isinstance(k, (list, set, tuple)):
        try:
            return [evals[n] for n in k]
        except IndexError:
            raise ValueError("given indices do not exist: ", n)
    elif k != None:
        return evals[: k]
    else:
        return evals

def eigenvectors(T, k=None, right=True):
    r"""Compute eigenvectors of given transition matrix.

    Eigenvectors are computed using the numpy.linalg interface 
    for the corresponding LAPACK routines.    

    Input
    -----
    T : numpy.ndarray, shape(d,d)
        Transition matrix (stochastic matrix).
    k : int (optional) or tuple of ints
        Compute the first k eigenvalues of T.

    Returns
    -------
    eigvec : numpy.ndarray, shape=(d, n)
        The eigenvectors of T ordered with decreasing absolute value of
        the corresponding eigenvalue. If k is None then n=d, if k is\
        int then n=k otherwise n is the length of the given tuple of\
        eigenvector indices.

    """
    if right:
        val, R=eig(T, left=False, right=True)
        """ Sorted eigenvalues and left and right eigenvectors. """
        perm=np.argsort(np.abs(val))[::-1]

        eigval=val[perm]
        eigvec=R[:,perm]        

    else:
        val, L  = eig(T, left=True, right=False)

        """ Sorted eigenvalues and left and right eigenvectors. """
        perm=np.argsort(np.abs(val))[::-1]

        eigval=val[perm]
        eigvec=L[:,perm]

    """ Return eigenvectors """
    if k==None:
        return eigvec
    elif isinstance(k, int):
        return eigvec[:,0:k]
    else:
        ind=np.asarray(k)
        return eigvec[:, ind] 
