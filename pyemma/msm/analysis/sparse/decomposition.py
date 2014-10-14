r"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the sparse implementations for functions specified in msm.api. 
Matrices are represented by scipy.sparse matrices throughout this module.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse.linalg

from scipy.sparse import eye
from scipy.sparse.linalg import factorized

import warnings

from pyemma.util.numeric import isclose
from pyemma.util.exceptions import ImaginaryEigenValueWarning, SpectralWarning

def backward_iteration(A, mu, x0, tol=1e-15, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) scipy.sparse matrix
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerace parameter for termination of iteration

    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu

    """
    T=A-mu*eye(A.shape[0])
    T=T.tocsc()
    """Prefactor T and return a function for solution"""
    solve=factorized(T)
    """Starting iterate with ||y_0||=1"""
    r0=1.0/np.linalg.norm(x0)
    y0=x0*r0
    """Local variables for inverse iteration"""
    y=1.0*y0
    r=1.0*r0
    N=0
    for iter in range(maxiter):
        x=solve(y)
        r=1.0/np.linalg.norm(x)
        y=x*r
        if r<=tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" %(maxiter, r)
    raise RuntimeError(msg)

def stationary_distribution_from_backward_iteration(P, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.

    Parameters
    ----------
    P : (M, M) scipy.sparse matrix
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.
        
    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A=P.transpose()
    mu=1.0-eps
    x0=np.ones(P.shape[0])
    y=backward_iteration(A, mu, x0)
    pi=y/y.sum()
    return pi

def stationary_distribution_from_eigenvector(T, ncv=None):
    r"""Compute stationary distribution of stochastic matrix T. 
      
    The stationary distribution is the left eigenvector corresponding to the 1
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns:
    --------
    mu : numpy array, shape(d,)      
        Vector of stationary probabilities.

    """
    vals, vecs=scipy.sparse.linalg.eigs(T.transpose(), k=1, which='LR', ncv=ncv)
    nu=vecs[:, 0].real
    mu=nu/np.sum(nu)
    return mu

def eigenvalues(T, k=None, ncv=None):
    r"""Compute the eigenvalues of a sparse transition matrix

    The first k eigenvalues of largest magnitude are computed.

    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    k : int (optional)
        Number of eigenvalues to compute.
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
    
    Returns
    -------
    v : ndarray
        Eigenvalues

    """
    if k is None:
        raise ValueError("Number of eigenvalues required for decomposition of sparse matrix")
    else:
        v=scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv)
        ind=np.argsort(np.abs(v))[::-1]
        return v[ind]
    
def eigenvectors(T, k=None, right=True, ncv=None):
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
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

        
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
            val, vecs=scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
            ind=np.argsort(np.abs(val))[::-1]
            return vecs[:,ind]
        else:            
            val, vecs=scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM', ncv=ncv)
            ind=np.argsort(np.abs(val))[::-1]
            return vecs[:, ind]        

def rdl_decomposition(T, k=None, norm='standard', ncv=None):
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
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k
        
    Returns
    -------
    R : (M, M) ndarray
        The normalized ("unit length") right eigenvectors, such that the 
        column ``R[:,i]`` is the right eigenvector corresponding to the eigenvalue 
        ``w[i]``, ``dot(T,R[:,i])``=``w[i]*R[:,i]``
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity    
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the 
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``    
      
    """
    if k is None:
        raise ValueError("Number of eigenvectors required for decomposition of sparse matrix")
    if norm=='standard':
        v, R=scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
        r, L=scipy.sparse.linalg.eigs(T.transpose(), k=k, which='LM', ncv=ncv)

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

        """Diagonal matrix with eigenvalues"""
        D=np.diag(v)

        return R, D, np.transpose(L)

    elif norm=='reversible':
        v, R=scipy.sparse.linalg.eigs(T, k=k, which='LM', ncv=ncv)
        mu=stationary_distribution_from_backward_iteration(T)

        """Sort right eigenvectors"""
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]
        R=R[:,ind]

        """Ensure that R[:,0] is positive"""
        R[:,0]=R[:,0]/np.sign(R[0,0])
        
        """Diagonal matrix with eigenvalues"""
        D=np.diag(v)      
        
        """Compute left eigenvectors from right ones"""
        L=mu[:, np.newaxis]*R        
        
        """Compute overlap"""
        s=np.diag(np.dot(np.transpose(L), R))
        
        """Renormalize left-and right eigenvectors to ensure L'R=Id"""
        R=R/np.sqrt(s[np.newaxis, :])
        L=L/np.sqrt(s[np.newaxis, :])           
        
        return R, D, np.transpose(L)              
    else:
        raise ValueError("Keyword 'norm' has to be either 'standard' or 'reversible'")
        
def timescales(T, tau=1, k=None, ncv=None):
    r"""Compute implied time scales of given transition matrix
    
    Parameters
    ----------
    T : transition matrix
    tau : lag time
    k : int (optional)
        Compute the first k implied time scales.
    ncv : int (optional)
        The number of Lanczos vectors generated, `ncv` must be greater than k;
        it is recommended that ncv > 2*k

    Returns
    -------
    ts : ndarray
        The implied time scales of the transition matrix.
    """
    if k is None:
        raise ValueError("Number of time scales required for decomposition of sparse matrix")    
    values=scipy.sparse.linalg.eigs(T, k=k, which='LM', return_eigenvectors=False, ncv=ncv)
    
    """Sort by absolute value"""
    ind=np.argsort(np.abs(values))[::-1]
    values=values[ind]
    
    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(values.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part '
                      'for implied time scale computation', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one=isclose(np.abs(values), 1.0)
    if sum(ind_abs_one)>1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts=np.zeros(len(values))

    """Eigenvalues of magnitude one imply infinite rate"""
    ts[ind_abs_one]=np.inf

    """All other eigenvalues give rise to finite rates"""
    ts[np.logical_not(ind_abs_one)]=\
        -1.0*tau/np.log(np.abs(values[np.logical_not(ind_abs_one)]))
    return ts

def timescales_from_eigenvalues(eval, tau=1):
    r"""Compute implied time scales from given eigenvalues
    
    Parameters
    ----------
    eval : eigenvalues
    tau : lag time

    Returns
    -------
    ts : ndarray
        The implied time scales to the given eigenvalues, in the same order.
    
    """
    
    """Check for dominant eigenvalues with large imaginary part"""
    if not np.allclose(eval.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part '
                      'for implied time scale computation', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one=isclose(np.abs(eval), 1.0)
    if sum(ind_abs_one)>1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts=np.zeros(len(eval))

    """Eigenvalues of magnitude one imply infinite timescale"""
    ts[ind_abs_one]=np.inf

    """All other eigenvalues give rise to finite timescales"""
    ts[np.logical_not(ind_abs_one)]=\
        -1.0*tau/np.log(np.abs(eval[np.logical_not(ind_abs_one)]))
    return ts
