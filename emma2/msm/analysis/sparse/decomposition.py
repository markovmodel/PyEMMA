"""This module provides matrix-decomposition based functions for the
analysis of stochastic matrices

Below are the sparse implementations for functions specified in msm.api. 
Dense matrices are represented by scipy.sparse matrices throughout this module.
"""

import numpy as np
import scipy.sparse.linalg

def mu(T):
    r"""Compute stationary distribution of stochastic matrix T. 
      
    The stationary distribution is the left eigenvector corresponding to the 
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns:
    --------
    mu : numpy array, shape(d,)      
        Vector of stationary probabilities.

    """
    vals, vecs=scipy.sparse.linalg.eigs(T.transpose(), k=1, which='LR')
    nu=vecs[:, 0].real
    mu=nu/sum(nu)
    return mu

