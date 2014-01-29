'''
Created on Jan 13, 2014

@author: noe
'''

import numpy as np

def transition_matrix_non_reversible(C):
    r"""
    Estimates a nonreversible transition matrix from count matrix C
    
    T_ij = c_ij / c_i where c_i = sum_j c_ij
     
    Parameters
    ----------
    C: ndarray, shape (n,n)
        count matrix
    
    Returns
    -------
    T: Estimated transition matrix
    
    """
    # multiply by 1.0 to make sure we're not doing integer division 
    rowsums = 1.0*np.sum(C,axis=1)
    if np.min(rowsums) <= 0:
        raise ValueError("Transition matrix has row sum of "+str(np.min(rowsums))+". Must have strictly positive row sums.")
    return np.divide(C, rowsums[:,np.newaxis])


def transition_matrix_reversible_fixpi(Z, mu, maxerr=1e-10, maxiter=10000, return_iterations = False):
    r"""
    maximum likelihood transition matrix with fixed stationary distribution
    
    developed by Fabian Paul and Frank Noe
    
    Parameters
    ----------
    Z: ndarray, shape (n,n)
        count matrix
    mu: ndarray, shape (n)
        stationary distribution
    maxerr: float
        Will exit (as converged) when the 2-norm of the Langrange multiplier vector changes less than maxerr
        in one iteration
    maxiter: int
        Will exit when reaching maxiter iterations without reaching convergence.
    return_iterations: bool (False)
        set true in order to return (T, it), where T is the transition matrix and it is the number of iterations needed
        
    Returns
    -------
    T, the transition matrix. When return_iterations=True, (T,it) is returned with it the number of iterations needed
    
    """
    it = 0
    n = len(mu)
    # constants
    B = Z + Z.transpose()
    # variables
    csum=np.sum(Z, axis=1)
    if (np.min(csum) <= 0):
        raise ValueError('Count matrix has rowsum(s) of zero. Require a count matrix with positive rowsums.')
    if (np.min(mu) <= 0):
        raise ValueError('Stationary distribution has zero elements. Require a positive stationary distribution.')
    l = 1.0*csum
    lnew = 1.0*csum
    q = np.zeros((n))
    A = np.zeros((n,n))
    D = np.zeros((n,n))
    # iterate lambda
    converged = False
    while (not converged) and (it < maxiter):
        # q_i = mu_i / l_i
        np.divide(mu, l, q)
        # d_ij = (mu_i / mu_j) * (l_j/l_i) + 1
        D[:] = q[:,np.newaxis]
        D /= q
        D += 1
        # a_ij = b_ij / d_ij
        np.divide(B, D, A)
        # new l_i = rowsum_i(A)
        np.sum(A, axis=1, out=lnew)
        # evaluate change
        err = np.linalg.norm(l-lnew,2)
        # is it converged?
        converged = (err <= maxerr)
        # copy new to old l-vector
        l[:] = lnew[:]
        it += 1
    if (not converged) and (it >= maxiter):
        raise ValueError('NOT CONVERGED: 2-norm of Langrange multiplier vector is still '
                    +str(err)+' > '+str(maxerr)+' after '+str(it)+' iterations. Increase maxiter or decrease maxerr')
    # compute T from Langrangian multipliers
    T = np.divide(A,l[:,np.newaxis])
    # return
    if return_iterations:
        return T, it
    else:
        return T

