"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the sparse implementations for functions specified in msm.api. 
Dense matrices are represented by scipy.sparse matrices throughout this module.
"""

import numpy as np
import scipy.sparse.linalg

def mu(T):
    r"""Compute stationary distribution of stochastic matrix T. 
      
    The stationary distribution is the left eigenvector corresponding to the 1
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
    vals, vecs=scipy.sparse.linalg.eigs(T.transpose(), k=1, which='LR')
    nu=vecs[:, 0].real
    mu=nu/sum(nu)
    return mu

def eigenvalues(T, k=None):
    r"""Compute the eigenvalues of a sparse transition matrix

    The first k eigenvalues of largest magnitude are computed.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    k : int (optional)
        Number of eigenvalues to compute.
    
    Returns
    -------
    v : ndarray
        Eigenvalues

    """
    if k is None:
        raise ValueError("Number of eigenvalues required for decomposition of sparse matrix")
    else:
        v=scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False)
        ind=np.argsort(np.abs(v))[::-1]
        return v[ind]
    
def eigenvectors(T, k=None, right=True):
    r"""Compute eigenvectors of given transition matrix.

    Eigenvectors are computed using the scipy interface 
    to the corresponding ARPACK routines.    

    Input
    -----
    T : scipy.sparse matrix
        Transition matrix (stochastic matrix).
    k : int (optional) or array-like 
        For integer k compute the first k eigenvalues of T
        else return those eigenvector sepcified by integer indices in k.
        
    Returns
    -------
    eigvec : numpy.ndarray, shape=(d, n)
        The eigenvectors of T ordered with decreasing absolute value of
        the corresponding eigenvalue. If k is None then n=d, if k is
        int then n=k otherwise n is the length of the given indices array.
        
    """
    if k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    else:
        if right:
            val, vecs=scipy.sparse.linalg.eigs(T, k=k, which='LM')
            ind=np.argsort(np.abs(val))[::-1]
            return vecs[:,ind]
        else:            
            val, vecs=scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM')
            ind=np.argsort(np.abs(val))[::-1]
            return vecs[:, ind]        
