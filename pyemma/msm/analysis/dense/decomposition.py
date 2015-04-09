r"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the dense implementations for functions specified in msm.api. 
Dense matrices are represented by numpy.ndarrays throughout this module.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import warnings

from scipy.linalg import eig, eigvals, solve, lu_factor, lu_solve
from pyemma.util.exceptions import SpectralWarning, ImaginaryEigenValueWarning
from pyemma.util.numeric import isclose

def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) ndarray
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
    T=A-mu*np.eye(A.shape[0])
    """LU-factor of T"""
    lupiv=lu_factor(T)
    """Starting iterate with ||y_0||=1"""
    r0=1.0/np.linalg.norm(x0)
    y0=x0*r0
    """Local variables for inverse iteration"""
    y=1.0*y0
    r=1.0*r0
    for i in range(maxiter):
        x=lu_solve(lupiv, y)
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
    P : (M, M) ndarray
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.
        
    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A=np.transpose(P)
    mu=1.0-eps
    x0=np.ones(P.shape[0])
    y=backward_iteration(A, mu, x0)
    pi=y/y.sum()
    return pi

def stationary_distribution_from_eigenvector(T):
    r"""Compute stationary distribution of stochastic matrix T. 

    The stationary distribution is the left eigenvector corresponding to the 
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    val, L = eig(T, left=True, right=False)

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
    evals=eigvals(T)
    """Sort by decreasing absolute value"""
    ind=np.argsort(np.abs(evals))[::-1]
    evals=evals[ind]

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

        #eigval=val[perm]
        eigvec=R[:,perm]        

    else:
        val, L  = eig(T, left=True, right=False)

        """ Sorted eigenvalues and left and right eigenvectors. """
        perm=np.argsort(np.abs(val))[::-1]

        #eigval=val[perm]
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
        auto: will be reversible if T is reversible, otherwise standard.

    Returns
    -------
    R : (M, M) ndarray
        The normalized (with respect to L) right eigenvectors, such that the 
        column R[:,i] is the right eigenvector corresponding to the eigenvalue 
        w[i], dot(T,R[:,i])=w[i]*R[:,i]
    D : (M, M) ndarray
        A diagonal matrix containing the eigenvalues, each repeated
        according to its multiplicity
    L : (M, M) ndarray
        The normalized (with respect to `R`) left eigenvectors, such that the 
        row ``L[i, :]`` is the left eigenvector corresponding to the eigenvalue
        ``w[i]``, ``dot(L[i, :], T)``=``w[i]*L[i, :]``
        
    """
    d=T.shape[0]
    w, R=eig(T)

    """Sort by decreasing magnitude of eigenvalue"""
    ind=np.argsort(np.abs(w))[::-1]
    w=w[ind]
    R=R[:,ind]   

    """Diagonal matrix containing eigenvalues"""
    D=np.diag(w)
    
    # auto-set norm
    if norm=='auto':
        from pyemma.msm.analysis import is_reversible
        if (is_reversible(T)):
            norm = 'reversible'
        else:
            norm = 'standard'
    # Standard norm: Euclidean norm is 1 for r and LR = I.
    if norm =='standard':
        L=solve(np.transpose(R), np.eye(d))
        
        """l1- normalization of L[:, 0]"""
        R[:, 0]=R[:, 0]*np.sum(L[:, 0])
        L[:, 0]=L[:, 0]/np.sum(L[:, 0])        

        if k is None:
            return R, D, np.transpose(L)
        else:
            return R[:,0:k], D[0:k,0:k], np.transpose(L[:,0:k])

    # Reversible norm:
    elif norm=='reversible':
        b=np.zeros(d)
        b[0]=1.0 

        A=np.transpose(R)
        nu=solve(A, b)
        mu=nu/np.sum(nu)

        """Ensure that R[:,0] is positive"""
        R[:,0]=R[:,0]/np.sign(R[0,0])

        """Use mu to connect L and R"""
        L=mu[:, np.newaxis]*R

        """Compute overlap"""
        s=np.diag(np.dot(np.transpose(L), R))

        """Renormalize left-and right eigenvectors to ensure L'R=Id"""
        R=R/np.sqrt(s[np.newaxis, :])
        L=L/np.sqrt(s[np.newaxis, :])

        if k is None:
            return R, D, np.transpose(L)
        else:
            return R[:,0:k], D[0:k,0:k], np.transpose(L[:,0:k])
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

    """Compute implied time scales"""
    return timescales_from_eigenvalues(values, tau)


def timescales_from_eigenvalues(evals, tau=1):
    r"""Compute implied time scales from given eigenvalues
    
    Parameters
    ----------
    evals : eigenvalues
    tau : lag time

    Returns
    -------
    ts : ndarray
        The implied time scales to the given eigenvalues, in the same order.
    
    """
    
    """Check for dominant eigenvalues with large imaginary part"""

    if not np.allclose(evals.imag, 0.0):
        warnings.warn('Using eigenvalues with non-zero imaginary part', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one=isclose(np.abs(evals), 1.0)
    if sum(ind_abs_one)>1:
        warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts=np.zeros(len(evals))

    """Eigenvalues of magnitude one imply infinite timescale"""
    ts[ind_abs_one]=np.inf

    """All other eigenvalues give rise to finite timescales"""
    ts[np.logical_not(ind_abs_one)]=\
        -1.0*tau/np.log(np.abs(evals[np.logical_not(ind_abs_one)]))
    return ts
    


