"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the sparse implementations for functions specified in msm.api. 
Dense matrices are represented by scipy.sparse matrices throughout this module.
"""

import numpy as np
import scipy.sparse.linalg

def stationary_distribution_from_eigenvector(T):
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
    mu=nu/np.sum(nu)
    return mu


def stationary_distribution_from_linearsystem(T):
    r"""Compute stationary distribution of stochastic matrix T. 
      
    The stationary distribution is the normalized solution of the System (T-I)x = 0.

    Input:
    ------
    T : scipy sparse array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns:
    --------
    mu : numpy array, shape(d,)      
        Vector of stationary probabilities.

    """
    n = scipy.shape(T)[0]
    # A = T' - I
    A = T.tolil().transpose() - scipy.sparse.eye(n)
    # b = 0
    b = scipy.sparse.lil_matrix((n,1))
    # Add constraint x_1 = 1 to first row
    A[0,0] += 1.0
    b[0,0] = 1.0
    # solve
    x = scipy.sparse.linalg.spsolve(A, b)
    # normalize
    x /= scipy.sum(x)
    # return 
    return x


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

def rdl_decomposition(T, k=None, norm='standard'):
    r"""Compute the decomposition into left and right eigenvectors.
    
    Parameters
    ----------
    T : sparse matrix 
        Transition matrix    
    k : int (optional)
        Number of eigenvector/eigenvalue pairs
    norm: {'standard', 'reversible'}
        standard: (L'R) = Id, L[:,0] is a probability distribution,
            the stationary distribution mu of T. Right eigenvectors
            R have a 2-norm of 1.
        reversible: R and L are related via L=L[:,0]*R.
        
    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, each repeated according to its multiplicity
    L : (M, M) ndarray
        The normalized ("unit length") left eigenvectors, such that the 
        column L[:,i] is the left eigenvector corresponding to the eigenvalue
        w[i], dot(L[:,i], T)=w[i]*L[:,i], L[:,0] is a probability distribution
        ("positive and l1 unit length").
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the 
        column R[:,i] is the right eigenvector corresponding to the eigenvalue 
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
      
    """
    if k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    if norm=='standard':
        v, R=scipy.sparse.linalg.eigs(T, k=k, which='LM')
        r, L=scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM')

        """Sort right eigenvectors"""
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]
        R=R[:,ind]

        """Sort left eigenvectors"""
        ind=np.argsort(np.abs(r))[::-1]
        r=r[ind]
        L=L[:,ind]        
        
        """l1-normalization of L[:, 0]"""
        L[:, 0]=L[:, 0]/np.sum(L[:, 0])
        
        """Standard normalization L'R=Id"""
        ov=np.diag(np.dot(np.transpose(L), R))
        R=R/ov[np.newaxis, :]

        return v, L, R

    elif norm=='reversible':
        v, R=scipy.sparse.linalg.eigs(T, k=k, which='LM')
        r, L=scipy.sparse.linalg.eigs(T.transpose(), k=1, which='LM')
        nu=L[:, 0]

        """Sort right eigenvectors"""
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]
        R=R[:,ind]

        mu=nu/np.sum(nu)
        L=mu[:, np.newaxis]*R        

        L[:, 0]=mu
        ov=np.diag(np.dot(np.transpose(L), R))
        R=R/ov[np.newaxis, :]

        return v, L, R        
        
    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")
        
def timescales(T, tau=1, k=None):
    r"""Compute implied time scales of given transition matrix
    
    Parameters
    ----------
    T : transition matrix
    tau : lag time
    k : int (optional)
        Compute the first k implied time scales.

    Returns
    -------
    ts : ndarray
        The implied time scales of the transition matrix.          
    
    """
    if k is None:
        raise ValueError("Number of time scales required for decomposition of sparse matrix")    
    values=scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False)
    
    """Sort by absolute value"""
    ind=np.argsort(np.abs(values))[::-1]
    values=values[ind]
    
    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(values.imag, 0.0):
        raise RuntimeWarning('Using eigenvalues with non-zero imaginary part '+\
                                     'for implied time scale computation')

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one=np.isclose(np.abs(values), 1.0)
    if sum(ind_abs_one)>1:
        raise RuntimeWarning('Multiple eigenvalues with magnitude one.')

    """Compute implied time scales"""
    ts=np.zeros(len(values))

    """Eigenvalues of magnitude one imply infinite rate"""
    ts[ind_abs_one]=np.inf

    """All other eigenvalues give rise to finite rates"""
    ts[np.logical_not(ind_abs_one)]=\
        -1.0*tau/np.log(np.abs(values[np.logical_not(ind_abs_one)]))
    return ts


