r"""Sparse implementation of mean first passage time computation

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

def mfpt(T, target):
    r"""Compute vector of mean first passage times to given target state.

    The vector m_t of mean first passage times to a given target state t 
    solves the following system of linear equations,

                0                       i=t
        m_t[i]=
                1+\sum_j p_ij m_t[j]    i \neq t


    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix.
    target : int
        Target state for mfpt calculation.

    Returns
    -------
    m_t : ndarray, shape=(n,)
        Vector of mean first passage times to target state t.
    
    """
    dim=T.shape[0]
    A=eye(dim, dim)-T

    """Convert to DOK (dictionary of keys) matrix to enable
    row-slicing and assignement"""
    A=A.todok()
    A[target, :]=0.0
    A[target, target]=1.0
    """Convert back to CSR-format for fast sparse linear algebra"""
    A=A.tocsr()

    b=np.ones(dim)
    b[target]=0.0
    m_t=spsolve(A, b)
    return m_t

