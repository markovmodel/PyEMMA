
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

r"""Unit test for the prior module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest
import warnings

import numpy as np
from pyemma.util.numeric import assert_allclose

from scipy.sparse import csr_matrix

from pyemma.util.numeric import allclose_sparse
from pyemma.msm.estimation import prior_neighbor, prior_const, prior_rev


class TestPriorDense(unittest.TestCase):
    def setUp(self):
        C = np.array([[4, 4, 0, 2], [4, 4, 1, 0], [0, 1, 4, 4], [0, 0, 4, 4]])
        self.C = C

        self.alpha_def = 0.001
        self.alpha = -0.5

        B_neighbor = np.array([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
        B_const = np.ones_like(C)
        B_rev = np.triu(B_const)

        self.B_neighbor = B_neighbor
        self.B_const = B_const
        self.B_rev = B_rev

    def tearDown(self):
        pass

    def test_prior_neighbor(self):
        Bn = prior_neighbor(self.C)
        assert_allclose(Bn, self.alpha_def * self.B_neighbor)

        Bn = prior_neighbor(self.C, alpha=self.alpha)
        assert_allclose(Bn, self.alpha * self.B_neighbor)

    def test_prior_const(self):
        Bn = prior_const(self.C)
        assert_allclose(Bn, self.alpha_def * self.B_const)

        Bn = prior_const(self.C, alpha=self.alpha)
        assert_allclose(Bn, self.alpha * self.B_const)

    def test_prior_rev(self):
        Bn = prior_rev(self.C)
        assert_allclose(Bn, -1.0 * self.B_rev)

        Bn = prior_rev(self.C, alpha=self.alpha)
        assert_allclose(Bn, self.alpha * self.B_rev)


class TestPriorSparse(unittest.TestCase):
    def setUp(self):
        C = np.array([[4, 4, 0, 2], [4, 4, 1, 0], [0, 1, 4, 4], [0, 0, 4, 4]])
        self.C = csr_matrix(C)

        self.alpha_def = 0.001
        self.alpha = -0.5

        B_neighbor = np.array([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
        B_const = np.ones_like(C)
        B_rev = np.triu(B_const)

        self.B_neighbor = csr_matrix(B_neighbor)
        self.B_const = B_const
        self.B_rev = B_rev

    def tearDown(self):
        pass

    def test_prior_neighbor(self):
        Bn = prior_neighbor(self.C)
        self.assertTrue(allclose_sparse(Bn, self.alpha_def * self.B_neighbor))

        Bn = prior_neighbor(self.C, alpha=self.alpha)
        self.assertTrue(allclose_sparse(Bn, self.alpha * self.B_neighbor))

    def test_prior_const(self):
        with warnings.catch_warnings(record=True) as w:
            Bn = prior_const(self.C)
            assert_allclose(Bn, self.alpha_def * self.B_const)

        with warnings.catch_warnings(record=True) as w:
            Bn = prior_const(self.C, alpha=self.alpha)
            assert_allclose(Bn, self.alpha * self.B_const)

    def test_prior_rev(self):
        with warnings.catch_warnings(record=True) as w:
            Bn = prior_rev(self.C)
            assert_allclose(Bn, -1.0 * self.B_rev)

        with warnings.catch_warnings(record=True) as w:
            Bn = prior_rev(self.C, alpha=self.alpha)
            assert_allclose(Bn, self.alpha * self.B_rev)


if __name__ == "__main__":
    unittest.main()