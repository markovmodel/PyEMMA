# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
    Sparse assessment module of msm analysis package
"""

from scipy.sparse.csgraph import connected_components
from scipy.sparse.sputils import isdense
from scipy.sparse.construct import diags

from pyemma.util.numeric import allclose_sparse

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
    T = T.tocsr()  # compressed sparse row for fast row slicing
    values = T.data  # non-zero entries of T

    """Check entry-wise positivity"""
    is_positive = np.allclose(values, np.abs(values), rtol=tol)

    """Check row normalization"""
    is_normed = np.allclose(T.sum(axis=1), 1.0, rtol=tol)

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
    K = K.tocsr()

    # check rows sum up to zero.
    row_sum = K.sum(axis=1)
    sum_eq_zero = np.allclose(row_sum, np.zeros(shape=row_sum.shape), atol=tol)

    # store copy of original diagonal
    org_diag = K.diagonal()

    # substract diagonal
    K = K - diags(org_diag, 0)

    # check off diagonals are > 0
    values = K.data
    values_gt_zero = np.allclose(values, np.abs(values), atol=tol)

    # add diagonal
    K = K + diags(org_diag, 0)

    return values_gt_zero and sum_eq_zero


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
    if not is_transition_matrix(T, tol):
        raise ValueError("given matrix is not a valid transition matrix.")

    T = T.tocsr()

    if mu is None:
        from decomposition import stationary_distribution_from_backward_iteration as statdist

        mu = statdist(T)

    Mu = diags(mu, 0)
    prod = Mu * T

    return allclose_sparse(prod, prod.transpose(), rtol=tol)


def is_connected(T, directed=True):
    r"""Check connectivity of the transition matrix.

    Return true, if the input matrix is completely connected,
    effectively checking if the number of connected components equals one.
    
    Parameters
    ----------
    T : scipy.sparse matrix 
        Transition matrix
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.       

    Returns
    -------
    connected : boolean, returning true only if T is connected.
        

    """
    nc = connected_components(T, directed=directed, connection='strong', \
                              return_labels=False)
    return nc == 1


def is_ergodic(T, tol):
    """
    checks if T is 'ergodic'
    
    Parameters
    ----------
    T : scipy.sparse matrix
        Transition matrix
    tol : float
        tolerance
        
    Returns
    -------
    Truth value : bool
    True, if # strongly connected components = 1
    False, otherwise
    """
    if isdense(T):
        T = T.tocsr()
    if not is_transition_matrix(T, tol):
        raise ValueError("given matrix is not a valid transition matrix.")

    num_components = connected_components(T, directed=True, \
                                          connection='strong', \
                                          return_labels=False)

    return num_components == 1