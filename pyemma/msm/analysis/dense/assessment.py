
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

import numpy as np

import decomposition


def is_transition_matrix(T, tol=1e-10):
    """
    Tests whether T is a transition matrix

    Parameters
    ----------
    T : ndarray shape=(n, n)
        matrix to test
    tol : float
        tolerance to check with

    Returns
    -------
    Truth value : bool
        True, if all elements are in interval [0, 1] 
            and each row of T sums up to 1.
        False, otherwise
    """
    if T.ndim != 2:
        return False
    if T.shape[0] != T.shape[1]:
        return False
    dim = T.shape[0]
    X = np.abs(T) - T
    x = np.sum(T, axis=1)
    return np.abs(x - np.ones(dim)).max() < dim * tol and X.max() < 2.0 * tol


def is_rate_matrix(K, tol=1e-10):
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

    row_sum = K.sum(axis=1)
    row_sum_eq_0 = np.allclose(row_sum, 0.0, atol=tol)

    return off_diagonal_positive and row_sum_eq_0


def is_reversible(T, mu=None, tol=1e-10):
    r"""
    checks whether T is reversible in terms of given stationary distribution.
    If no distribution is given, it will be calculated out of T.

    It performs following check:
    :math:`\pi_i P_{ij} = \pi_j P_{ji}`

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
        True, if T is a reversible transitition matrix
        False, otherwise
    """
    if is_transition_matrix(T, tol):
        if mu is None:
            mu = decomposition.stationary_distribution_from_backward_iteration(T)
        X = mu[:, np.newaxis] * T
        return np.allclose(X, np.transpose(X),  atol=tol)
    else:
        raise ValueError("given matrix is not a valid transition matrix.")