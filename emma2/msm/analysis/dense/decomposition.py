"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the dense implementations for functions specified in msm.api. 
Dense matrices are represented by numpy.ndarrays throughout this module.
"""

import numpy as np
from scipy.linalg import eig, eigvals, solve

def stationary_distribution(T):
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


def rdl_decomposition(T, k=None, norm='standard'):
    r"""Compute the decomposition into left and right eigenvectors.
    
    Parameters
    ----------
    T : (M, M) ndarray 
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
    d=T.shape[0]
    w, R=eig(T)

    """Sort by decreasing magnitude of eigenvalue"""
    ind=np.argsort(np.abs(w))[::-1]
    w=w[ind]
    R=R[:,ind]   

    if norm =='standard':
        L=solve(np.transpose(R), np.eye(d))
        
        """l1- normalization of L[:, 0]"""
        R[:, 0]=R[:, 0]*np.sum(L[:, 0])
        L[:, 0]=L[:, 0]/np.sum(L[:, 0])
        

        if k is None:
            return w, L, R
        else:
            return w[0:k], L[:,0:k], R[:,0:k]

    elif norm=='reversible':
        b=np.zeros(d)
        b[0]=1.0 

        A=np.transpose(R)
        nu=solve(A, b)
        mu=nu/np.sum(nu)

        """Make the first right eigenvector the constant one vector"""
        R[:, 0]=R[:, 0]*np.sum(nu)

        """Use mu to connect L and R"""
        L=mu[:, np.newaxis]*R

        """Compute overlap"""
        ov=np.diag(np.dot(np.transpose(L), R))

        """Renormalize the left eigenvectors to ensure L'R=Id"""
        L=L/ov[np.newaxis, :]

        if k is None:
            return w, L, R
        else:
            return w[0:k], L[:,0:k], R[:,0:k]
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
    values=eigvals(T)

    """Sort by absolute value"""
    ind=np.argsort(np.abs(values))[::-1]
    values=values[ind]
    
    if k is None:
        values=values
    else:
        values=values[0:k]

    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(values.imag, 0.0):
        raise RuntimeWarning('Using eigenvalues with non-zero imaginary part'+\
                                     'for implied time scale computation')

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one=np.abs(values)==1.0
    if sum(ind_abs_one)>1:
        raise RuntimeWarning('Multiple eigenvalues with magnitude one.')

    """Compute implied time scales"""
    ts=np.zeros(len(values))

    """Eigenvalues of magnitude one imply infinite rate"""
    ts[ind_abs_one]=inf

    """All other eigenvalues give rise to finite rates"""
    ts[np.logical_not(ind_abs_one)]=-1.0*tau/log(np.abs(values[np.logical_not(ind_abs_one)]))
    return ts
        
   
    
