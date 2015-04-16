
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

r"""Test package for the decomposition module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from scipy.linalg import eig, eigvals

from birth_death_chain import BirthDeathChain

from decomposition import stationary_distribution_from_eigenvector
from decomposition import stationary_distribution_from_backward_iteration
from decomposition import eigenvalues, eigenvectors, rdl_decomposition
from decomposition import timescales


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10
        self.ncv = 40

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim / 2 - 1] = 0.001
        q[self.dim / 2 + 1] = 0.001

        self.bdc = BirthDeathChain(q, p)

    def test_statdist_decomposition(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_eigenvector(P, ncv=self.ncv)
        assert_allclose(mu, mun)

    def test_statdist_iteration(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_backward_iteration(P)
        assert_allclose(mu, mun)

    def test_eigenvalues(self):
        P = self.bdc.transition_matrix()
        P_dense = self.bdc.transition_matrix()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]

        """k=None"""
        with self.assertRaises(ValueError):
            evn = eigenvalues(P)

        """k is not None"""
        evn = eigenvalues(P, k=self.k)
        assert_allclose(ev[0:self.k], evn)

        """k is not None and ncv is not None"""
        evn = eigenvalues(P, k=self.k, ncv=self.ncv)
        assert_allclose(ev[0:self.k], evn)

    def test_eigenvectors(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        ev, L, R = eig(P_dense, left=True, right=True)
        ind = np.argsort(np.abs(ev))[::-1]
        ev = ev[ind]
        R = R[:, ind]
        L = L[:, ind]
        vals = ev[0:self.k]

        """k=None"""
        with self.assertRaises(ValueError):
            Rn = eigenvectors(P)

        with self.assertRaises(ValueError):
            Ln = eigenvectors(P, right=False)

        """k is not None"""
        Rn = eigenvectors(P, k=self.k)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """k is not None and ncv is not None"""
        Rn = eigenvectors(P, k=self.k, ncv=self.ncv)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, ncv=self.ncv)
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

    def test_rdl_decomposition(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln = rdl_decomposition(P)

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """k is not None, ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """Reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln = rdl_decomposition(P, norm='reversible')

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, norm='reversible')
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        """k is not None ncv is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k, norm='reversible', ncv=self.ncv)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(P.dot(Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

    def test_timescales(self):
        P_dense = self.bdc.transition_matrix()
        P = self.bdc.transition_matrix_sparse()
        ev = eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        ts = -1.0 / np.log(np.abs(ev))

        """k=None"""
        with self.assertRaises(ValueError):
            tsn = timescales(P)

        """k is not None"""
        tsn = timescales(P, k=self.k)
        assert_allclose(ts[1:self.k], tsn[1:])

        """k is not None, ncv is not None"""
        tsn = timescales(P, k=self.k, ncv=self.ncv)
        assert_allclose(ts[1:self.k], tsn[1:])

        """tau=7"""

        """k is not None"""
        tsn = timescales(P, k=self.k, tau=7)
        assert_allclose(7 * ts[1:self.k], tsn[1:])


if __name__ == "__main__":
    unittest.main()

    
    
    
    
    