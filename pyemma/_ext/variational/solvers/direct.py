from __future__ import absolute_import
import numpy as _np
from ..util import ZeroRankError as _ZeroRankError

__author__ = 'noe'


def sort_by_norm(evals, evecs):
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
    evnorms = _np.abs(evals)
    # sort
    I = _np.argsort(evnorms)[::-1]
    # permute
    evals2 = evals[I]
    evecs2 = evecs[:, I]
    # done
    return evals2, evecs2


def spd_eig(W, epsilon=1e-10, method='QR', canonical_signs=False):
    """ Rank-reduced eigenvalue decomposition of symmetric positive definite matrix.

    Removes all negligible eigenvalues

    Parameters
    ----------
    W : ndarray((n, n), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W
    canonical_signs : boolean, default = False
        Fix signs in V, s. t. the largest element of in every row of V is positive.

    Returns
    -------
    s : ndarray(k)
        k non-negligible eigenvalues, sorted by descending norms

    V : ndarray(n, k)
        k leading eigenvectors
    """
    # check input
    assert _np.allclose(W.T, W), 'W is not a symmetric matrix'

    if method.lower() == 'qr':
        from .eig_qr.eig_qr import eig_qr
        s, V = eig_qr(W)
    # compute the Eigenvalues of C0 using Schur factorization
    elif method.lower() == 'schur':
        from scipy.linalg import schur
        S, V = schur(W)
        s = _np.diag(S)
    else:
        raise ValueError('method not implemented: ' + method)

    s, V = sort_by_norm(s, V) # sort them

    # determine the cutoff. We know that C0 is an spd matrix,
    # so we select the truncation threshold such that everything that is negative vanishes
    evmin = _np.min(s)
    if evmin < 0:
        epsilon = max(epsilon, -evmin + 1e-14)

    # determine effective rank m and perform low-rank approximations.
    evnorms = _np.abs(s)
    n = _np.shape(evnorms)[0]
    m = n - _np.searchsorted(evnorms[::-1], epsilon)
    if m == 0:
        raise _ZeroRankError('All eigenvalues are smaller than %g, rank reduction would discard all dimensions.'%epsilon)
    Vm = V[:, 0:m]
    sm = s[0:m]

    if canonical_signs:
        # enforce canonical eigenvector signs
        for j in range(m):
            jj = _np.argmax(_np.abs(Vm[:, j]))
            Vm[:, j] *= _np.sign(Vm[jj, j])

    return sm, Vm


def spd_inv(W, epsilon=1e-10, method='QR'):
    """
    Compute matrix inverse of symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd
    (Moore-Penrose inverse).

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

    Returns
    -------
    L : ndarray((n, r))
        the Moore-Penrose inverse of the symmetric positive-definite matrix :math:`W`

    """
    if (_np.shape(W)[0] == 1):
        if W[0,0] < epsilon:
            raise _ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        Winv = 1./W[0,0]
    else:
        sm, Vm = spd_eig(W, epsilon=epsilon, method=method)
        Winv = _np.dot(Vm, _np.diag(1.0 / sm)).dot(Vm.T)

    # return split
    return Winv


def spd_inv_sqrt(W, epsilon=1e-10, method='QR', return_rank=False):
    """
    Computes :math:`W^{-1/2}` of symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd.

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

    Returns
    -------
    L : ndarray((n, r))
        :math:`W^{-1/2}` after reduction of W to a low-rank spd matrix

    """
    if _np.shape(W)[0] == 1:
        if W[0,0] < epsilon:
            raise _ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        Winv = 1./_np.sqrt(W[0, 0])
        sm = _np.ones(1)
    else:
        sm, Vm = spd_eig(W, epsilon=epsilon, method=method)
        Winv = _np.dot(Vm, _np.diag(1.0 / _np.sqrt(sm))).dot(Vm.T)

    # return split
    if return_rank:
        return Winv, sm.shape[0]
    else:
        return Winv


def spd_inv_split(W, epsilon=1e-10, method='QR', canonical_signs=False):
    """
    Compute :math:`W^{-1} = L L^T` of the symmetric positive-definite matrix :math:`W`.

    by first reducing W to a low-rank approximation that is truly spd.

    Parameters
    ----------
    W : ndarray((m,m), dtype=float)
        Symmetric positive-definite (spd) matrix.
    epsilon : float
        Truncation parameter. Eigenvalues with norms smaller than this cutoff will
        be removed.
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W

     canonical_signs : boolean, default = False
        Fix signs in L, s. t. the largest element of in every row of L is positive.

    Returns
    -------
    L : ndarray((n, r))
        Matrix :math:`L` from the decomposition :math:`W^{-1} = L L^T`.

    """
    if (_np.shape(W)[0] == 1):
        if W[0,0] < epsilon:
            raise _ZeroRankError(
                'All eigenvalues are smaller than %g, rank reduction would discard all dimensions.' % epsilon)
        L = 1./_np.sqrt(W[0,0])
    else:
        sm, Vm = spd_eig(W, epsilon=epsilon, method=method, canonical_signs=canonical_signs)
        L = _np.dot(Vm, _np.diag(1.0/_np.sqrt(sm)))

    # return split
    return L


def eig_corr(C0, Ct, epsilon=1e-10, method='QR', sign_maxelement=False):
    r""" Solve generalized eigenvalue problem with correlation matrices C0 and Ct

    Numerically robust solution of a generalized Hermitian (symmetric) eigenvalue
    problem of the form

    .. math::
        \mathbf{C}_t \mathbf{r}_i = \mathbf{C}_0 \mathbf{r}_i l_i

    Computes :math:`m` dominant eigenvalues :math:`l_i` and eigenvectors
    :math:`\mathbf{r}_i`, where :math:`m` is the numerical rank of the problem.
    This is done by first conducting a Schur decomposition of the symmetric
    positive matrix :math:`\mathbf{C}_0`, then truncating its spectrum to
    retain only eigenvalues that are numerically greater than zero, then using
    this decomposition to define an ordinary eigenvalue Problem for
    :math:`\mathbf{C}_t` of size :math:`m`, and then solving this eigenvalue
    problem.

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
    method : str
        Method to perform the decomposition of :math:`W` before inverting. Options are:

        * 'QR': QR-based robust eigenvalue decomposition of W
        * 'schur': Schur decomposition of W
    sign_maxelement : bool
        If True, re-scale each eigenvector such that its entry with maximal absolute value
        is positive.


    Returns
    -------
    l : ndarray (m)
        The first m generalized eigenvalues, sorted by descending norm
    R : ndarray (n,m)
        The first m generalized eigenvectors, as a column matrix.

    """
    L = spd_inv_split(C0, epsilon=epsilon, method=method, canonical_signs=True)
    Ct_trans = _np.dot(_np.dot(L.T, Ct), L)

    # solve the symmetric eigenvalue problem in the new basis
    if _np.allclose(Ct.T, Ct):
        from scipy.linalg import eigh
        l, R_trans = eigh(Ct_trans)
    else:
        from scipy.linalg import eig
        l, R_trans = eig(Ct_trans)

    # sort eigenpairs
    l, R_trans = sort_by_norm(l, R_trans)

    # transform the eigenvectors back to the old basis
    R = _np.dot(L, R_trans)

    # Change signs of eigenvectors:
    if sign_maxelement:
        for j in range(R.shape[1]):
            imax = _np.argmax(_np.abs(R[:, j]))
            R[:, j] *= _np.sign(R[imax, j])

    # return result
    return l, R
