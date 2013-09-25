from numpy import *

def check_nonnegativity(A):
    """Checks nonnegativity of a matrix.

    Matrix A=(:math:`a_{ij}`) is nonnegative if
    :math:`a_{ij} \geq 0` for all :math:`i, j`.

    Parameters
    ----------
    A : ndarray, shape=(M, N)
        The matrix to test.

    Returns
    -------
    nonnegative : bool
        The truth value of the nonnegativity test.

    Notes
    -----
    The nonnegativity test is performed using
    boolean ndarrays.

    Nonnegativity is import for transition matrix estimation.

    Examples
    --------
    >>> import numpy as np
    >>> A=np.array([[0.4, 0.1, 0.4], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
    >>> x=check_nonnegativity(A)
    >>> x
    True
    
    >>> B=np.array([[1.0, 0.0], [2.0, 3.0]])
    >>> x=check_nonnegativity(A)
    >>> x
    False

    """
    ind=(A<=0.0)
    return sum(ind)==0
    
    
