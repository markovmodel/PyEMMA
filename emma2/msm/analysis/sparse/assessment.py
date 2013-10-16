import numpy as np
from scipy.sparse.csr import isspmatrix_csr, csr_matrix
from scipy.sparse.lil import lil_matrix

from scipy.sparse.csgraph import connected_components
from scipy.sparse.sputils import isdense

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
    r"""
    checks whether T is reversible in terms of given stationary distribution.
    If no distribution is given, it will be calculated out of T.
    
    performs follwing check:
    :math:`\pi_i P_{ij} = \pi_j P_{ji}
    Parameters
    ----------
    T : scipy.sparse matrix
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
    if is_transition_matrix(T, tol):
        # todo test: csr supports slicing (lil does)
        if isinstance(T, (csr_matrix, lil_matrix)):
            return np.allclose(T * mu[ : , np.newaxis ], \
                           T[ : , np.newaxis] * mu,  atol=tol)
        else:
            r = T * mu
            return np.allclose(r, np.transpose(r), atol=tol)
    else:
        ValueError("given matrix is not a valid transition matrix.")
        
def is_ergodic(T, tol):
    if isdense(T):
        T = csr_matrix(T)
    num_components = connected_components(T, directed=True, \
                                          connection='strong', \
                                          return_labels=False)
    
    return num_components == 1