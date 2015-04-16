
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

r"""Unit tests for the covariance module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from covariance import tmatrix_cov, dirichlet_covariance, error_perturbation


class TestCovariance(unittest.TestCase):
    def setUp(self):
        alpha1 = np.array([1.0, 2.0, 1.0])
        cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2 = np.array([2.0, 1.0, 2.0])
        cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C = np.zeros((3, 3))
        self.C[0, :] = alpha1 - 1.0
        self.C[1, :] = alpha2 - 1.0
        self.C[2, :] = alpha1 - 1.0

        self.cov = np.zeros((3, 3, 3))
        self.cov[0, :, :] = cov1
        self.cov[1, :, :] = cov2
        self.cov[2, :, :] = cov1

    def tearDown(self):
        pass

    def test_tmatrix_cov(self):
        cov = tmatrix_cov(self.C)
        assert_allclose(cov, self.cov)

        cov = tmatrix_cov(self.C, row=1)
        assert_allclose(cov, self.cov[1, :, :])


class TestDirichletCovariance(unittest.TestCase):
    def setUp(self):
        self.alpha1 = np.array([1.0, 2.0, 1.0])
        self.cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        self.alpha2 = np.array([2.0, 1.0, 2.0])
        self.cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

    def tearDown(self):
        pass

    def test_dirichlet_covariance(self):
        cov = dirichlet_covariance(self.alpha1)
        assert_allclose(cov, self.cov1)

        cov = dirichlet_covariance(self.alpha2)
        assert_allclose(cov, self.cov2)


class TestErrorPerturbation(unittest.TestCase):
    def setUp(self):
        alpha1 = np.array([1.0, 2.0, 1.0])
        cov1 = 1.0 / 80 * np.array([[3.0, -2.0, -1.0], [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2 = np.array([2.0, 1.0, 2.0])
        cov2 = 1.0 / 150 * np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C = np.zeros((3, 3))
        self.C[0, :] = alpha1 - 1.0
        self.C[1, :] = alpha2 - 1.0
        self.C[2, :] = alpha1 - 1.0

        self.cov = np.zeros((3, 3, 3))
        self.cov[0, :, :] = cov1
        self.cov[1, :, :] = cov2
        self.cov[2, :, :] = cov1

        """Scalar-valued observable f(P)=P_11+P_22+P_33"""
        self.S1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        """Vector-valued observable f(P)=(P_11, P_12)"""
        self.S2 = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])

        """Error-perturbation scalar observable"""
        self.x = (self.S1[:, :, np.newaxis] * self.cov * self.S1[:, np.newaxis, :]).sum()

        """Error-perturbation vector observable"""
        tmp = self.S2[:, np.newaxis, :, :, np.newaxis] * self.cov[np.newaxis, np.newaxis, :, :, :] * \
              self.S2[np.newaxis, :, :, np.newaxis, :]
        self.X = np.sum(tmp, axis=(2, 3, 4))

    def tearDown(self):
        pass

    def test_error_perturbation(self):
        xn = error_perturbation(self.C, self.S1)
        assert_allclose(xn, self.x)

        Xn = error_perturbation(self.C, self.S2)
        assert_allclose(Xn, self.X)


if __name__ == "__main__":
    unittest.main()