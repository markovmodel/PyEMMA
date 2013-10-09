import numpy as np


def is_stochastic_matrix(T, tol=1e-15):
    dim = T.shape[0]
    X = np.abs(T) - T
    x = np.sum(T, axis = 1)
    return np.abs(x - np.ones(dim)).max() < dim * tol and X.max() < 2.0 * tol


def is_rate_matrix(K, tol=1e-15):
    """
    True if K is a rate matrix
    Parameters
    ----------
    K : numpy.ndarray matrix
        Matrix to check
    tol : float
        tolerance to check with
        
    Returns
    -------
    Truth value : bool
        True, if K negated diagonal is positive and row sums up to zero.
        False, otherwise
    """
    R = K - K.diagonal()
    off_diagonal_positive = np.allclose(R, abs(R), 0.0, atol=tol)
    
    row_sum = K.sum(axis = 1)
    row_sum_eq_0 = np.allclose(row_sum, 0.0, atol=tol)
    
    return off_diagonal_positive and row_sum_eq_0

def is_reversible(T, mu=None, tol):
    r"""
        checks whether T is reversible in terms of given stationary distribution.
        If no distribution is given, it will be calculated out of T.
        
        performs follwing check:
        :math:`\pi_i P_{ij} = \pi_j P_{ji}
        Parameters
        ----------
        T : numpy.ndarray matrix
            Transition matrix
        mu : numpy.ndarray vector
            stationary distribution
        tol : float
            tolerance to check with
            
        Returns
        -------
        Truth value : bool
            True, if T is a stochastic matrix
            False, otherwise
    """
    if is_stochastic_matrix(T, tol):
        if mu is None:
            mu = mu(T)
        return np.allclose(T * mu[ : , np.newaxis ], \
                           T[ : , np.newaxis] * mu,  atol=tol)
    else:
        raise ValueError("given matrix is not a valid transition matrix.")