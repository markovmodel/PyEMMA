
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

r"""Test package for the expectations module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
import scipy
import scipy.linalg
from pyemma.util.numeric import diags, choice
import scipy.sparse
import scipy.sparse.linalg

import expectations


def random_orthonormal_sparse_vectors(d, k):
    r"""Generate a random set of k orthonormal sparse vectors 

    The algorithm draws random indices, {i_1,...,i_k}, from the set
    of all possible indices, {0,...,d-1}, without replacement.
    Random sparse vectors v are given by

    v[i]=k^{-1/2} for i in {i_1,...,i_k} and zero elsewhere.

    """
    indices = choice(d, replace=False, size=(k * k))
    indptr = np.arange(0, k * (k + 1), k)
    values = 1.0 / np.sqrt(k) * np.ones(k * k)
    return scipy.sparse.csc_matrix((values, indices, indptr))


def sparse_allclose(A, B, rtol=1e-5, atol=1e-08):
    r"""Perform the allclose test for two sparse matrices

    Parameters
    ----------
    A : (M, N) sparse matrix
    B : (M, N) sparse matrix

    Returns
    -------
    allclose: bool
        The truth value of the sparse allclose test.

    """
    A = A.tocsr()
    B = B.tocsr()

    allclose_values = np.allclose(A.data, B.data, rtol=rtol, atol=atol)
    equal_indices = np.array_equal(A.indices, B.indices)
    equal_indptr = np.array_equal(A.indptr, B.indptr)

    return allclose_values and equal_indices and equal_indptr


class TestExpectedCounts(unittest.TestCase):
    def setUp(self):
        self.k = 20
        self.d = 10000
        """Generate a random kxk dense transition matrix"""
        C = np.random.random_integers(0, 100, size=(self.k, self.k))
        C = C + np.transpose(C)  # Symmetric count matrix for real eigenvalues
        T = 1.0 * C / np.sum(C, axis=1)[:, np.newaxis]
        v, L, R = scipy.linalg.eig(T, left=True, right=True)
        """Sort eigenvalues and eigenvectors, order is decreasing absolute value"""
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        L = L[:, ind]
        R = R[:, ind]

        nu = L[:, 0]
        mu = nu / np.sum(nu)

        """
        Generate k random sparse
        orthorgonal vectors of dimension d
        """
        Q = random_orthonormal_sparse_vectors(self.d, self.k)

        """Push forward dense decomposition to sparse one via Q"""
        self.L_sparse = Q.dot(scipy.sparse.csr_matrix(L))
        self.R_sparse = Q.dot(scipy.sparse.csr_matrix(R))
        self.v_sparse = v  # Eigenvalues are invariant

        """Push forward transition matrix and stationary distribution"""
        self.T_sparse = Q.dot(scipy.sparse.csr_matrix(T)).dot(Q.transpose())
        self.mu_sparse = Q.dot(mu) / np.sqrt(self.k)

    def tearDown(self):
        pass

    def test_expected_counts(self):
        N = 50
        T = self.T_sparse
        p0 = self.mu_sparse

        EC_n = expectations.expected_counts(p0, T, N)

        D_mu = diags(self.mu_sparse, 0)
        EC_true = N * D_mu.dot(T)

        self.assertTrue(sparse_allclose(EC_true, EC_n))


class TestExpectedCountsStationary(unittest.TestCase):
    def setUp(self):
        self.k = 20
        self.d = 10000
        """Generate a random kxk dense transition matrix"""
        C = np.random.random_integers(0, 100, size=(self.k, self.k))
        C = C + np.transpose(C)  # Symmetric count matrix for real eigenvalues
        T = 1.0 * C / np.sum(C, axis=1)[:, np.newaxis]
        v, L, R = scipy.linalg.eig(T, left=True, right=True)
        """Sort eigenvalues and eigenvectors, order is decreasing absolute value"""
        ind = np.argsort(np.abs(v))[::-1]
        v = v[ind]
        L = L[:, ind]
        R = R[:, ind]

        nu = L[:, 0]
        mu = nu / np.sum(nu)

        """
        Generate k random sparse
        orthorgonal vectors of dimension d
        """
        Q = random_orthonormal_sparse_vectors(self.d, self.k)

        """Push forward dense decomposition to sparse one via Q"""
        self.L_sparse = Q.dot(scipy.sparse.csr_matrix(L))
        self.R_sparse = Q.dot(scipy.sparse.csr_matrix(R))
        self.v_sparse = v  # Eigenvalues are invariant

        """Push forward transition matrix and stationary distribution"""
        self.T_sparse = Q.dot(scipy.sparse.csr_matrix(T)).dot(Q.transpose())
        self.mu_sparse = Q.dot(mu) / np.sqrt(self.k)

    def tearDown(self):
        pass

    def test_expected_counts_stationary(self):
        n = 50
        T = self.T_sparse
        mu = self.mu_sparse

        D_mu = diags(self.mu_sparse, 0)
        EC_true = n * D_mu.dot(T)

        """Compute mu on the fly"""
        EC_n = expectations.expected_counts_stationary(T, n)
        self.assertTrue(sparse_allclose(EC_true, EC_n))

        """With precomputed mu"""
        EC_n = expectations.expected_counts_stationary(T, n, mu=mu)
        self.assertTrue(sparse_allclose(EC_true, EC_n))

        n = 0
        EC_true = scipy.sparse.coo_matrix(T.shape)
        EC_n = expectations.expected_counts_stationary(T, n)
        self.assertTrue(sparse_allclose(EC_true, EC_n))


if __name__ == "__main__":
    unittest.main()