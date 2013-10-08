import numpy as np
  
def is_transition_matrix(T, tol):
    """
    True if T is a transition matrix
    
    Parameters
    ----------
    T : scipy.sparse matrix
        Matrix to check
    tol : float
        tolerance to check with
    
    Returns
    -------
    Truth value: bool
        True, if T is positive and normed
        False, otherwise
    
    """
    T=T.tocsr() # compressed sparse row for fast row slicing
    values=T.data # non-zero entries of T

    """Check entry-wise positivity"""
    is_positive=np.allclose(values, np.abs(values), rtol=0.0, atol=tol)

    """Check row normalization"""
    is_normed=np.allclose(T.sum(axis=1), 1.0, rtol=0.0, atol=tol)

    return is_positive and is_normed


def is_rate_matrix(K, tol):
    """
    True if K is a rate matrix
    Parameters
    ----------
    K : scipy.sparse matrix
        Matrix to check
    tol : float
        tolerance to check with
        
    Returns
    -------
    Truth value : bool
        True, if K negated diagonal is positive and row sums up to zero.
        False, otherwise
    """
    values = K.data
    # store copy of original diagonal
    org_diag = K.diagonal().copy()
    diag = K.diagonal()
    # set diagonal to 0
    diag[:] = 0
    
    # check all values are greater zero within given tolerance
    gt_zero = np.allclose(values-values, 0.0, atol = tol)
    # restore original diagonal
    diag = org_diag
    
    return gt_zero


def is_reversible(T, mu=None, tol=1e-15):
    r"""True if T is a transition matrix
        mu : tests with respect to this stationary distribution
    """
    if is_transition_matrix(T, tol):
        raise NotImplementedError("not yet impled for sparse.")
    else:
        ValueError("given matrix is not a valid transition matrix.")
