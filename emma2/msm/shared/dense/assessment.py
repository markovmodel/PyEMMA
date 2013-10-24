import numpy as np

def is_transition_matrix(T, tol=1e-15):
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
    off_diagonal_positive = np.allclose(R, abs(R), 0.0, tol)
    
    row_sum = K.sum(axis = 1)
    row_sum_eq_0 = np.allclose(row_sum, 0.0, rtol=0.0, atol=tol)
    
    return off_diagonal_positive and row_sum_eq_0