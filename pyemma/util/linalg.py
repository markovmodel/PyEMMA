__author__ = 'noe'

import numpy as np
import scipy.linalg


def _sort_by_norm(evals, evecs):
    """
    Sorts the eigenvalues and eigenvectors by descending norm of the eigenvalues

    Parameters
    ----------
    evals: ndarray(n)
        eigenvalues
    evecs: ndarray(n,n)
        eigenvectors in a column matrix

    Returns
    -------
    (evals, evecs) : ndarray(m), ndarray(n,m)
        the sorted eigenvalues and eigenvectors

    """
    # norms
    evnorms = np.abs(evals)
    # sort
    I = np.argsort(evnorms)[::-1]
    # permute
    evals2 = evals[I]
    evecs2 = evecs[:, I]
    # done
    return (evals2, evecs2)


def eig_corr(C0, Ct, epsilon=1e-6):
    """
    Solve the generalized eigenvalues problem with correlation matrices C0 and Ct

    Parameters
    ----------
    C0 : ndarray (n,n)
        time-instantaneous correlation matrix. Must be symmetric positive definite
    Ct : ndarray (n,n)
        time-lagged correlation matrix. Must be symmetric
    epsilon : float
        eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
        cut off. The remaining number of Eigenvalues define the size of
        the output.

    Returns
    -------
    l : ndarray (m)
        The first m generalized eigenvalues, sorted by descending norm
    R : ndarray (n,m)
        The first m generalized eigenvectors, as a column matrix.

    """
    # check input
    assert np.allclose(C0.T, C0), 'C0 is not a symmetric matrix'
    assert np.allclose(Ct.T, Ct), 'Ct is not a symmetric matrix'

    # compute the Eigenvalues of C0 using Schur factorization
    (S, V) = scipy.linalg.schur(C0)
    s = np.diag(S)
    (s, V) = _sort_by_norm(s, V) # sort them
    S = np.diag(s)

    # determine the cutoff. We know that C0 is an spd matrix,
    # so we select the truncation threshold such that everything that is negative vanishes
    evmin = np.min(s)
    if evmin < 0:
        epsilon = max(epsilon, -evmin + 1e-16)

    # determine effective rank m and perform low-rank approximations.
    evnorms = np.abs(s)
    n = np.shape(evnorms)[0]
    m = n - np.searchsorted(evnorms[::-1], epsilon)
    Vm = V[:, 0:m]
    sm = s[0:m]

    # transform Ct to orthogonal basis given by the eigenvectors of C0
    Sinvhalf = 1.0 / np.sqrt(sm)
    T = np.dot(np.diag(Sinvhalf), Vm.T)
    Ct_trans = np.dot(np.dot(T, Ct), T.T)

    # solve the symmetric eigenvalue problem in the new basis
    (l, R_trans) = scipy.linalg.eigh(Ct_trans)
    (l, R_trans) = _sort_by_norm(l, R_trans)

    # transform the eigenvectors back to the old basis
    R = np.dot(T.T, R_trans)

    # return result
    return (l, R)
