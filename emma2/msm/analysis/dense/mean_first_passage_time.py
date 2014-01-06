"""Dense implementation of mean first passage time computations"""
import numpy as np
import scipy.linalg

def mfpt(T, target):
    r"""Compute vector of mean first passage times to given target state.

    The vector m_t of mean first passage times to a given target state t 
    solves the following system of linear equations,

                0                       i=t
        m_t[i]=    
                1+\sum_j p_ij m_t[j]    i \neq t
        
    
    Parameters
    ----------
    T : ndarray, shape=(n,n) 
        Transition matrix.
    target : int
        Target state for mfpt calculation.
    
    Returns
    -------
    m_t : ndarray, shape=(n,)
        Vector of mean first passage times to target state t.
    
    """
    dim=T.shape[0]
    A=np.eye(dim)-T
    A[target, :]=0.0
    A[target, target]=1.0
    b=np.ones(dim)
    b[target]=0.0
    m_t=scipy.linalg.solve(A, b)
    return m_t


    
