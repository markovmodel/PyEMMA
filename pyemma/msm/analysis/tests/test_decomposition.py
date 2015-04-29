
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

r"""Unit test for decomposition functions in api.py

.. moduleauthor:: Benjamin Trendelkamp-Schroer<benjamin DOT trendelkamp-schorer AT fu-berlin DOT de>

"""
import unittest
import warnings

import numpy as np
from pyemma.util.numeric import assert_allclose

from scipy.linalg import eig, eigvals
from scipy.sparse import csr_matrix

from pyemma.util.exceptions import SpectralWarning, ImaginaryEigenValueWarning
from birth_death_chain import BirthDeathChain

from pyemma.msm.analysis import stationary_distribution, eigenvalues, eigenvectors
from pyemma.msm.analysis import rdl_decomposition, timescales

################################################################################
# Dense
################################################################################


class TestDecompositionDense(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim / 2 - 1] = 0.001
        q[self.dim / 2 + 1] = 0.001

        self.bdc = BirthDeathChain(q, p)

    def test_statdist(self):
        P = self.bdc.transition_matrix()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution(P)
        assert_allclose(mu, mun)

    def test_eigenvalues(self):
        P = self.bdc.transition_matrix()
        ev = eigvals(P)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]

        """k=None"""
        evn = eigenvalues(P)
        assert_allclose(ev, evn)

        """k is not None"""
        evn = eigenvalues(P, k=self.k)
        assert_allclose(ev[0:self.k], evn)

    def test_eigenvectors(self):
        P = self.bdc.transition_matrix()

        # k==None
        ev = eigvals(P)
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        Dn = np.diag(ev)

        # right eigenvectors
        Rn = eigenvectors(P)
        assert_allclose(np.dot(P,Rn),np.dot(Rn,Dn))
        # left eigenvectors
        Ln = eigenvectors(P, right=False).T
        assert_allclose(np.dot(Ln.T,P),np.dot(Dn,Ln.T))
        # orthogonality
        Xn = np.dot(Ln.T, Rn)
        di = np.diag_indices(Xn.shape[0])
        Xn[di] = 0.0
        assert_allclose(Xn,0)

        # k!=None
        Dnk = Dn[:,0:self.k][0:self.k,:]
        # right eigenvectors
        Rn = eigenvectors(P, k=self.k)
        assert_allclose(np.dot(P,Rn),np.dot(Rn,Dnk))
        # left eigenvectors
        Ln = eigenvectors(P, right=False, k=self.k).T
        assert_allclose(np.dot(Ln.T,P),np.dot(Dnk,Ln.T))
        # orthogonality
        Xn = np.dot(Ln.T, Rn)
        di = np.diag_indices(self.k)
        Xn[di] = 0.0
        assert_allclose(Xn,0)

    def test_rdl_decomposition(self):
        P = self.bdc.transition_matrix()
        mu = self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        Rn, Dn, Ln = rdl_decomposition(P)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(np.dot(P, Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(np.dot(Ln, P), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.dim))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, k=self.k)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(np.dot(P, Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(np.dot(Ln, P), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)

        """Reversible"""

        """k=None"""
        Rn, Dn, Ln = rdl_decomposition(P, norm='reversible')
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(np.dot(P, Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(np.dot(Ln, P), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.dim))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

        """k is not None"""
        Rn, Dn, Ln = rdl_decomposition(P, norm='reversible', k=self.k)
        Xn = np.dot(Ln, Rn)
        """Right-eigenvectors"""
        assert_allclose(np.dot(P, Rn), np.dot(Rn, Dn))
        """Left-eigenvectors"""
        assert_allclose(np.dot(Ln, P), np.dot(Dn, Ln))
        """Orthonormality"""
        assert_allclose(Xn, np.eye(self.k))
        """Probability vector"""
        assert_allclose(np.sum(Ln[0, :]), 1.0)
        """Reversibility"""
        assert_allclose(Ln.transpose(), mu[:, np.newaxis] * Rn)

    def test_timescales(self):
        P = self.bdc.transition_matrix()
        ev = eigvals(P)
        """Sort with decreasing magnitude"""
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        ts = -1.0 / np.log(np.abs(ev))

        """k=None"""
        tsn = timescales(P)
        assert_allclose(ts[1:], tsn[1:])

        """k is not None"""
        tsn = timescales(P, k=self.k)
        assert_allclose(ts[1:self.k], tsn[1:])

        """tau=7"""

        """k=None"""
        tsn = timescales(P, tau=7)
        assert_allclose(7 * ts[1:], tsn[1:])

        """k is not None"""
        tsn = timescales(P, k=self.k, tau=7)
        assert_allclose(7 * ts[1:self.k], tsn[1:])


class TestTimescalesDense(unittest.TestCase):
    def setUp(self):
        self.T = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self.P = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
        self.W = np.array([[0, 1], [1, 0]])

    def test_timescales_1(self):
        """Multiple eigenvalues of magnitude one,
        eigenvalues with non-zero imaginary part"""
        ts = np.array([np.inf, np.inf])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsn = timescales(self.W)
            assert_allclose(tsn, ts)
            assert issubclass(w[-1].category, SpectralWarning)

    def test_timescales_2(self):
        """Eigenvalues with non-zero imaginary part"""
        ts = np.array([np.inf, 0.971044, 0.971044])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsn = timescales(0.5 * self.T + 0.5 * self.P)
            assert_allclose(tsn, ts)
            assert issubclass(w[-1].category, ImaginaryEigenValueWarning)

        ################################################################################


# Sparse
################################################################################

class TestDecompositionSparse(unittest.TestCase):
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

    def test_statdist(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution(P)
        assert_allclose(mu, mun)

    def test_eigenvalues(self):
        P = self.bdc.transition_matrix_sparse()
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

        Ln = eigenvectors(P, right=False, k=self.k).T
        assert_allclose(P.transpose().dot(Ln), vals[np.newaxis, :] * Ln)

        """k is not None and ncv is not None"""
        Rn = eigenvectors(P, k=self.k, ncv=self.ncv)
        assert_allclose(vals[np.newaxis, :] * Rn, P.dot(Rn))

        Ln = eigenvectors(P, right=False, k=self.k, ncv=self.ncv).T
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