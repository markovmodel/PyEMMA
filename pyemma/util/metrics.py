__author__ = 'noe'

import numpy as np
from pyemma.util.linalg import mdot

"""
Contains various metrics for ranking kinetic models
"""


def _svd_sym_koopman(K, C00_train, Ctt_train, k):
    """ Computes the SVD of the symmetrized Koopman operator in the empirical distribution.
    """
    from pyemma._ext.variational.solvers.direct import spd_inv_sqrt
    # reweight operator to empirical distribution
    C0t_re = mdot(C00_train, K)
    # symmetrized operator and SVD
    K_sym = mdot(spd_inv_sqrt(C00_train), C0t_re, spd_inv_sqrt(Ctt_train))
    if k is not None:
        from scipy.sparse.linalg import svds
        U, S, Vt = svds(K_sym, k=k)
    else:
        U, S, Vt = np.linalg.svd(K_sym, compute_uv=True, full_matrices=False)
    # projects back to singular functions of K
    U = mdot(spd_inv_sqrt(C00_train), U)
    Vt = mdot(Vt,spd_inv_sqrt(Ctt_train))
    return U, S, Vt.T


def vamp_1_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=None):
    """ Computes the VAMP-1 score of a kinetic model.

    Ranks the kinetic model described by the estimation of covariances C00, C0t and Ctt,
    defined by:


        :math:`C_{0t}^{train} = E_t[x_t x_{t+\tau}^T]`
        :math:`C_{tt}^{train} = E_t[x_{t+\tau} x_{t+\tau}^T]`

    These model covariances might have been subject to symmetrization or reweighting,
    depending on the type of model used.

    The covariances C00, C0t and Ctt of the test data are direct empirical estimates.
    singular vectors U and V using the test data
    with covariances C00, C0t, Ctt. U and V should come from the SVD of the symmetrized
    transition matrix or Koopman matrix:

        :math:`(C00^{train})^{-(1/2)} C0t^{train} (Ctt^{train})^{-(1/2)}  = U S V.T`

    Parameters:
    -----------
    K : ndarray(n, k)
        left singular vectors of the symmetrized transition matrix or Koopman matrix
    C00_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{00}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_train : ndarray(n, n)
        time-lagged covariance matrix of the training data, defined by
        :math:`C_{0t}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{tt}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    C00_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{00}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_test : ndarray(n, n)
        time-lagged covariance matrix of the test data, defined by
        :math:`C_{0t}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{tt}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    k : int
        number of slow processes to consider in the score

    Returns:
    --------
    vamp1 : float
        VAMP-1 score

    """
    from pyemma._ext.variational.solvers.direct import spd_inv_sqrt

    # SVD of symmetrized operator in empirical distribution
    U, S, V = _svd_sym_koopman(K, C00_train, Ctt_train, k=k)
    A = spd_inv_sqrt(mdot(U.T, C00_test, U))
    B = mdot(U.T, C0t_test, V)
    C = spd_inv_sqrt(mdot(V.T, Ctt_test, V))

    # compute trace norm (nuclear norm), equal to the sum of singular values
    score = np.linalg.norm(mdot(A, B, C), ord='nuc')
    return score


def vamp_2_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=None):
    """ Computes the VAMP-2 score of a kinetic model.

    Ranks the kinetic model described by the estimation of covariances C00, C0t and Ctt,
    defined by:


        :math:`C_{0t}^{train} = E_t[x_t x_{t+\tau}^T]`
        :math:`C_{tt}^{train} = E_t[x_{t+\tau} x_{t+\tau}^T]`

    These model covariances might have been subject to symmetrization or reweighting,
    depending on the type of model used.

    The covariances C00, C0t and Ctt of the test data are direct empirical estimates.
    singular vectors U and V using the test data
    with covariances C00, C0t, Ctt. U and V should come from the SVD of the symmetrized
    transition matrix or Koopman matrix:

        :math:`(C00^{train})^{-(1/2)} C0t^{train} (Ctt^{train})^{-(1/2)}  = U S V.T`

    Parameters:
    -----------
    K : ndarray(n, k)
        left singular vectors of the symmetrized transition matrix or Koopman matrix
    C00_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{00}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_train : ndarray(n, n)
        time-lagged covariance matrix of the training data, defined by
        :math:`C_{0t}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{tt}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    C00_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{00}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_test : ndarray(n, n)
        time-lagged covariance matrix of the test data, defined by
        :math:`C_{0t}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{tt}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    k : int
        number of slow processes to consider in the score

    Returns:
    --------
    vamp2 : float
        VAMP-2 score

    """
    from pyemma._ext.variational.solvers.direct import spd_inv_sqrt

    # SVD of symmetrized operator in empirical distribution
    U, S, V = _svd_sym_koopman(K, C00_train, Ctt_train, k=k)
    A = spd_inv_sqrt(mdot(U.T, C00_test, U))
    B = mdot(U.T, C0t_test, V)
    C = spd_inv_sqrt(mdot(V.T, Ctt_test, V))

    # compute square frobenius, equal to the sum of squares of singular values
    score = np.linalg.norm(mdot(A, B, C), ord='fro') ** 2
    return score


def vamp_e_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=None):
    """ Computes the VAMP-E score of a kinetic model.

    Ranks the kinetic model described by the estimation of covariances C00, C0t and Ctt,
    defined by:


        :math:`C_{0t}^{train} = E_t[x_t x_{t+\tau}^T]`
        :math:`C_{tt}^{train} = E_t[x_{t+\tau} x_{t+\tau}^T]`

    These model covariances might have been subject to symmetrization or reweighting,
    depending on the type of model used.

    The covariances C00, C0t and Ctt of the test data are direct empirical estimates.
    singular vectors U and V using the test data
    with covariances C00, C0t, Ctt. U and V should come from the SVD of the symmetrized
    transition matrix or Koopman matrix:

        :math:`(C00^{train})^{-(1/2)} C0t^{train} (Ctt^{train})^{-(1/2)}  = U S V.T`

    Parameters:
    -----------
    K : ndarray(n, k)
        left singular vectors of the symmetrized transition matrix or Koopman matrix
    C00_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{00}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_train : ndarray(n, n)
        time-lagged covariance matrix of the training data, defined by
        :math:`C_{0t}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_train : ndarray(n, n)
        covariance matrix of the training data, defined by
        :math:`C_{tt}^{train} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    C00_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{00}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_t^T`
    C0t_test : ndarray(n, n)
        time-lagged covariance matrix of the test data, defined by
        :math:`C_{0t}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_t x_{t+\tau}^T`
    Ctt_test : ndarray(n, n)
        covariance matrix of the test data, defined by
        :math:`C_{tt}^{test} = (T-\tau)^{-1} \sum_{t=0}^{T-\tau} x_{t+\tau} x_{t+\tau}^T`
    k : int
        number of slow processes to consider in the score

    Returns:
    --------
    vampE : float
        VAMP-E score

    """
    # SVD of symmetrized operator in empirical distribution
    U, s, V = _svd_sym_koopman(K, C00_train, Ctt_train, k=k)
    S = np.diag(s[:k])
    score = np.trace(2.0 * mdot(V, S, U.T, C0t_test) - mdot(V, S, U.T, C00_test, U, S, V.T, Ctt_test))
    return score


def vamp_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=None, score='VAMP2'):
    if score.lower() == 'vamp1':
        return vamp_1_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=k)
    elif score.lower() == 'vamp2':
        return vamp_2_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=k)
    elif score.lower() == 'vampe':
        return vamp_e_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=k)
    else:
        raise ValueError('Unknown score: ' + str(score))
