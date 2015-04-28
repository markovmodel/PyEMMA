
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

'''
Created on 06.12.2013

@author: jan-hendrikprinz

This module provides the unittest for the pcca module

'''
import unittest
import numpy as np

from pyemma.util.numeric import assert_allclose
from pcca import pcca


class TestPCCA(unittest.TestCase):
    def test_pcca_no_transition_matrix(self):
        P = np.array([[1.0, 1.0],
                      [0.1, 0.9]])

        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_no_detailed_balance(self):
        P = np.array([[0.8, 0.1, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.6, 0.3, 0.1]])
        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_1(self):
        P = np.array([[1, 0],
                      [0, 1]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_2(self):
        P = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.999, 0.001],
                      [0.0, 0.001, 0.999]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=3
        chi = pcca(P, 3)
        sol = np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_4(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_5(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0.5, 0.5],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0.5, 0.5, 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_large(self):
        import os

        P = np.loadtxt(os.path.split(__file__)[0] + '/../tests/P_rev_251x251.dat')
        # n=2
        chi = pcca(P, 2)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=3
        chi = pcca(P, 3)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=4
        chi = pcca(P, 4)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))

    def test_pcca_coarsegrain(self):
        # fine-grained transition matrix
        P = np.array([[0.9,  0.1,  0.0,  0.0,  0.0],
                      [0.1,  0.89, 0.01, 0.0,  0.0],
                      [0.0,  0.1,  0.8,  0.1,  0.0],
                      [0.0,  0.0,  0.01, 0.79, 0.2],
                      [0.0,  0.0,  0.0,  0.2,  0.8]])
        from pyemma.msm.analysis import stationary_distribution
        pi = stationary_distribution(P)
        Pi = np.diag(pi)
        m = 3
        # Susanna+Marcus' expression ------------
        M = pcca(P, m)
        pi_c = np.dot(M.T, pi)
        Pi_c_inv = np.diag(1.0/pi_c)
        # restriction and interpolation operators
        R = M.T
        I = np.dot(np.dot(Pi, M), Pi_c_inv)
        # result
        ms1 = np.linalg.inv(np.dot(R,I)).T
        ms2 = np.dot(np.dot(I.T, P), R.T)
        Pc_ref = np.dot(ms1,ms2)
        # ---------------------------------------

        from pcca import coarsegrain
        Pc = coarsegrain(P, 3)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(Pc - Pc_ref)) < 1e-10
        # test mass conservation
        assert np.allclose(Pc.sum(axis=1), np.ones(m))

        from pcca import PCCA
        p = PCCA(P, m)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(p.coarse_grained_transition_matrix - Pc_ref)) < 1e-10
        # test against the present coarse-grained stationary dist
        assert np.max(np.abs(p.coarse_grained_stationary_probability - pi_c)) < 1e-10
        # test mass conservation
        assert np.allclose(p.coarse_grained_transition_matrix.sum(axis=1), np.ones(m))

if __name__ == "__main__":
    unittest.main()