import numpy as np
  
def is_transition_matrix(T, tol=1e-15):
    """
    True if T is a transition matrix
    
    Parameters
    ----------
    T : sipy.sparse matrix
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
