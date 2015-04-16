
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    rowsums = 1.0 * np.sum(C, axis=1)
    if np.min(rowsums) <= 0:
        raise ValueError(
            "Transition matrix has row sum of " + str(np.min(rowsums)) + ". Must have strictly positive row sums.")
    return np.divide(C, rowsums[:, np.newaxis])


def __initX(C):
    """
    Computes an initial guess for a reversible correlation matrix
    """
    from ..api import tmatrix
    from ...analysis import statdist

    T = tmatrix(C)
    mu = statdist(T)
    Corr = np.dot(np.diag(mu), T)
    return 0.5 * (Corr + Corr.T)


def __relative_error(x, y, norm=None):
    """
    computes the norm of the vector with elementwise relative errors 
    between x and y, defined by (x_i - y_i) / (x_i + y_i)

    x : ndarray (n)
        vector 1
    y : ndarray (n)
        vector 2
    norm : vector norm to be used. By default the Euclidean norm is used.
        This value is passed as 'ord' to numpy.linalg.norm()

    """
    d = (x - y)
    s = (x + y)
    # to avoid dividing by zero, always set to 0 
    nz = np.nonzero(d)
    # relative error vector
    erel = d[nz] / s[nz]
    # return euclidean norm
    return np.linalg.norm(erel, ord=norm)


def estimate_transition_matrix_reversible(C, Xinit=None, maxiter=1000000, maxerr=1e-8,
                                          return_statdist=False, return_conv=False):
    """
    iterative method for estimating a maximum likelihood reversible transition matrix
    
    The iteration equation implemented here is:
        t_ij = (c_ij + c_ji) / ((c_i / x_i) + (c_j / x_j))
    Please note that there is a better (=faster) iteration that has been described in
    Prinz et al, J. Chem. Phys. 134, p. 174105 (2011). We should implement that too.
    
    Parameters
    ----------
    C : ndarray (n,n)
        count matrix. If a non-connected count matrix is used, the method returns in error
    Xinit = None : ndarray (n,n)
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) T, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    maxerr = 1000000 : int
        maximum number of iterations before the method exits
    maxiter = 1e-8 : float
        convergence tolerance. This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (x_i = sum_k x_ik). The relative stationary probability changes
        e_i = (x_i^(1) - x_i^(2))/(x_i^(1) + x_i^(2)) are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, |e_i|_2, is compared to convtol.
    return_statdist = False : Boolean
        If set to true, the stationary distribution is also returned
    return_conv = False : Boolean
        If set to true, the likelihood history and the pi_change history is returned.

    Returns
    -------
    T or (T,pi) or (T,lhist,pi_changes) or (T,pi,lhist,pi_changes)
    T : ndarray (n,n)
        transition matrix. This is the only return for return_statdist = False, return_conv = False
    (pi) : ndarray (n)
        stationary distribution. Only returned if return_statdist = True
    (lhist) : ndarray (k)
        likelihood history. Has the length of the number of iterations needed. 
        Only returned if return_conv = True
    (pi_changes) : ndarray (k)
        history of likelihood history. Has the length of the number of iterations needed. 
        Only returned if return_conv = True
    """
    from ..api import is_connected
    from ...estimation import log_likelihood
    # check input
    if (not is_connected(C)):
        ValueError('Count matrix is not fully connected. ' +
                   'Need fully connected count matrix for ' +
                   'reversible transition matrix estimation.')
    converged = False
    n = np.shape(C)[0]
    # initialization
    C2 = C + C.T  # reversibly counted matrix
    nz = np.nonzero(C2)
    csum = np.sum(C, axis=1)  # row sums C
    X = Xinit
    if (X is None):
        X = __initX(C)  # initial X
    xsum = np.sum(X, axis=1)  # row sums x
    D = np.zeros((n, n))  # helper matrix
    T = np.zeros((n, n))  # transition matrix
    # if convergence history requested, initialize variables 
    if (return_conv):
        diffs = np.zeros(maxiter)
        # likelihood
        lhist = np.zeros(maxiter)
        T = X / xsum[:, np.newaxis]
        lhist[0] = log_likelihood(C, T)
    # iteration
    i = 1
    while (i < maxiter - 1) and (not converged):
        # c_i / x_i
        c_over_x = csum / xsum
        # d_ij = (c_i/x_i) + (c_j/x_j)
        D[:] = c_over_x[:, np.newaxis]
        D += c_over_x
        # update estimate
        X[nz] = C2[nz] / D[nz]
        X[nz] /= np.sum(X[nz])  # renormalize
        xsumnew = np.sum(X, axis=1)
        # compute difference in pi
        diff = __relative_error(xsum, xsumnew)
        # update pi
        xsum = xsumnew
        # any convergence history wanted?
        if (return_conv):
            # update T and likelihood
            T = X / xsum[:, np.newaxis]
            lhist[i] = log_likelihood(C, T)
            diffs[i] = diff
        # converged?
        converged = (diff < maxerr)
        i += 1
    # finalize and return
    T = X / xsum[:, np.newaxis]
    if (return_statdist and return_conv):
        return (T, xsum, lhist[0:i], diffs[0:i])
    if (return_statdist):
        return (T, xsum)
    if (return_conv):
        return (T, lhist[0:i], diffs[0:i])
    return T  # else just return T


def transition_matrix_reversible_fixpi(Z, mu, maxerr=1e-10, maxiter=10000, return_iterations=False):
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
    csum = np.sum(Z, axis=1)
    if (np.min(csum) <= 0):
        raise ValueError('Count matrix has rowsum(s) of zero. Require a count matrix with positive rowsums.')
    if (np.min(mu) <= 0):
        raise ValueError('Stationary distribution has zero elements. Require a positive stationary distribution.')
    if (np.min(np.diag(Z)) == 0):
        raise ValueError(
            'Count matrix has diagonals with 0. Cannot guarantee convergence of algorithm. Suggestion: add a small prior (e.g. 1e-10) to the diagonal')
    l = 1.0 * csum
    lnew = 1.0 * csum
    q = np.zeros((n))
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    # iterate lambda
    converged = False
    while (not converged) and (it < maxiter):
        # q_i = mu_i / l_i
        np.divide(mu, l, q)
        # d_ij = (mu_i / mu_j) * (l_j/l_i) + 1
        D[:] = q[:, np.newaxis]
        D /= q
        D += 1
        # a_ij = b_ij / d_ij
        np.divide(B, D, A)
        # new l_i = rowsum_i(A)
        np.sum(A, axis=1, out=lnew)
        # evaluate change
        err = np.linalg.norm(l - lnew, 2)
        # is it converged?
        converged = (err <= maxerr)
        # copy new to old l-vector
        l[:] = lnew[:]
        it += 1
    if (not converged) and (it >= maxiter):
        raise ValueError('NOT CONVERGED: 2-norm of Langrange multiplier vector is still '
                         + str(err) + ' > ' + str(maxerr) + ' after ' + str(
            it) + ' iterations. Increase maxiter or decrease maxerr')
    # compute T from Langrangian multipliers
    T = np.divide(A, l[:, np.newaxis])
    # return
    if return_iterations:
        return T, it
    else:
        return T