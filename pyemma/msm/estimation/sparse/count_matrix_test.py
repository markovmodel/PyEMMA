
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

"""Unit tests for the count_matrix module"""

import unittest

from os.path import abspath, join
from os import pardir

import numpy as np
from pyemma.util.numeric import assert_allclose
import scipy.sparse

from count_matrix import count_matrix, count_matrix_mult
from count_matrix import count_matrix_bincount, count_matrix_bincount_mult
from count_matrix import count_matrix_coo, count_matrix_coo_mult
from count_matrix import make_square_coo_matrix, add_coo_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'

################################################################################
# count_matrix
################################################################################

class TestCountMatrix(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.S_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.B1_lag = np.array([[1, 2], [2, 1]])
        self.B2_lag = np.array([[0, 1], [1, 1]])
        self.B3_lag = np.array([[2, 0], [0, 0]])

        self.B1_sliding = np.array([[1, 2], [2, 1]])
        self.B2_sliding = np.array([[1, 2], [1, 1]])
        self.B3_sliding = np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        self.S_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.C1_lag = np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C = count_matrix(self.S_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix(self.S_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix(self.S_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix(self.S_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix(self.S_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix(self.S_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix(self.S_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix(self.S_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix(self.S_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix(self.S_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix(self.S_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix(self.S_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix(self.S_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix(self.S_short, 1, nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = count_matrix(self.S_short, 1, nstates=1)


class TestCountMatrixtMult(unittest.TestCase):
    def setUp(self):
        M = 10
        self.M = M

        """Small test cases"""
        dtraj_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.dtrajs_short = [dtraj_short for i in range(M)]

        self.B1_lag = M * np.array([[1, 2], [2, 1]])
        self.B2_lag = M * np.array([[0, 1], [1, 1]])
        self.B3_lag = M * np.array([[2, 0], [0, 0]])

        self.B1_sliding = M * np.array([[1, 2], [2, 1]])
        self.B2_sliding = M * np.array([[1, 2], [1, 1]])
        self.B3_sliding = M * np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        dtraj_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.dtrajs_long = [dtraj_long for i in range(M)]
        self.C1_lag = M * np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = M * np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = M * np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = M * np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = M * np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = M * np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_count_matrix_mult(self):
        """Small test cases"""
        C = count_matrix_mult(self.dtrajs_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix_mult(self.dtrajs_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix_mult(self.dtrajs_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix_mult(self.dtrajs_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix_mult(self.dtrajs_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix_mult(self.dtrajs_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix_mult(self.dtrajs_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix_mult(self.dtrajs_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix_mult(self.dtrajs_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix_mult(self.dtrajs_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix_mult(self.dtrajs_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix_mult(self.dtrajs_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix_mult(self.dtrajs_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix_mult(self.dtrajs_short, 1, sliding=False, nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = count_matrix_mult(self.dtrajs_short, 1, sliding=False, nstates=1)


################################################################################
# coo
################################################################################

class TestCountMatrixCooMult(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.S1 = np.array([0, 0, 0, 1, 1, 1])
        self.S2 = np.array([0, 0, 0, 1, 1, 1])

        self.B1_sliding = np.array([[4, 2], [0, 4]])
        self.B2_sliding = np.array([[2, 4], [0, 2]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C = count_matrix_coo_mult([self.S1, self.S2], 1, sliding=True).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix_coo_mult([self.S1, self.S2], 2, sliding=True).toarray()
        assert_allclose(C, self.B2_sliding)

    def test_nstates_keyword(self):
        C = count_matrix_coo_mult([self.S1, self.S2], 1, sliding=True, nstates=10).toarray()
        self.assertTrue(C.shape == (10, 10))


class TestCountMatrixCoo(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.S_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.B1_lag = np.array([[1, 2], [2, 1]])
        self.B2_lag = np.array([[0, 1], [1, 1]])
        self.B3_lag = np.array([[2, 0], [0, 0]])

        self.B1_sliding = np.array([[1, 2], [2, 1]])
        self.B2_sliding = np.array([[1, 2], [1, 1]])
        self.B3_sliding = np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        self.S_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.C1_lag = np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = np.loadtxt(testpath + 'C_13_sliding.dat')


    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        C = count_matrix_coo(self.S_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix_coo(self.S_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix_coo(self.S_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix_coo(self.S_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix_coo(self.S_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix_coo(self.S_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix_coo(self.S_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix_coo(self.S_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix_coo(self.S_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix_coo(self.S_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix_coo(self.S_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix_coo(self.S_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix_coo(self.S_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix_coo(self.S_short, 1, nstates=10)
        self.assertTrue(C.shape == (10, 10))


class TestAddCooMatrix(unittest.TestCase):
    def setUp(self):
        An = np.array([[1.0, 2.0], [3.0, 4.0]])
        Bn = np.array([[-1.0, 3.0], [-4.0, 2.0]])
        Cn = np.array([[0.0, 5.0], [-1.0, 6.0]])

        self.A = scipy.sparse.coo_matrix(An)
        self.B = scipy.sparse.coo_matrix(Bn)
        self.C = scipy.sparse.coo_matrix(Cn)

    def tearDown(self):
        pass

    def test_add_coo_matrix(self):
        C_test = add_coo_matrix(self.A, self.B)
        assert_allclose(C_test.toarray(), self.C.toarray())


class TestMakeSquareCooMatrix(unittest.TestCase):
    def setUp(self):
        An = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.An_square = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
        self.A = scipy.sparse.coo_matrix(An)

        Bn = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.Bn_square = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
        self.B = scipy.sparse.coo_matrix(Bn)

    def tearDown(self):
        pass

    def test_make_square_coo_matrix(self):
        A_test = make_square_coo_matrix(self.A)
        assert_allclose(A_test.toarray(), self.An_square)

        B_test = make_square_coo_matrix(self.B)
        assert_allclose(B_test.toarray(), self.Bn_square)

    ################################################################################


# bincount
################################################################################

class TestCountMatrixBincount(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.S_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.B1_lag = np.array([[1, 2], [2, 1]])
        self.B2_lag = np.array([[0, 1], [1, 1]])
        self.B3_lag = np.array([[2, 0], [0, 0]])

        self.B1_sliding = np.array([[1, 2], [2, 1]])
        self.B2_sliding = np.array([[1, 2], [1, 1]])
        self.B3_sliding = np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        self.S_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.C1_lag = np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = np.loadtxt(testpath + 'C_13_sliding.dat')

    def tearDown(self):
        pass

    def test_count_matrix_bincount(self):
        """Small test cases"""
        C = count_matrix_bincount(self.S_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix_bincount(self.S_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix_bincount(self.S_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix_bincount(self.S_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix_bincount(self.S_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix_bincount(self.S_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix_bincount(self.S_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix_bincount(self.S_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix_bincount(self.S_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix_bincount(self.S_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix_bincount(self.S_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix_bincount(self.S_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix_bincount(self.S_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix_bincount(self.S_short, 1, nstates=10)
        self.assertTrue(C.shape == (10, 10))


class TestCountMatrixBincountMult(unittest.TestCase):
    def setUp(self):
        M = 10
        self.M = M

        """Small test cases"""
        dtraj_short = np.array([0, 0, 1, 0, 1, 1, 0])
        self.dtrajs_short = [dtraj_short for i in range(M)]

        self.B1_lag = M * np.array([[1, 2], [2, 1]])
        self.B2_lag = M * np.array([[0, 1], [1, 1]])
        self.B3_lag = M * np.array([[2, 0], [0, 0]])

        self.B1_sliding = M * np.array([[1, 2], [2, 1]])
        self.B2_sliding = M * np.array([[1, 2], [1, 1]])
        self.B3_sliding = M * np.array([[2, 1], [0, 1]])

        """Larger test cases"""
        dtraj_long = np.loadtxt(testpath + 'dtraj.dat').astype(int)
        self.dtrajs_long = [dtraj_long for i in range(M)]
        self.C1_lag = M * np.loadtxt(testpath + 'C_1_lag.dat')
        self.C7_lag = M * np.loadtxt(testpath + 'C_7_lag.dat')
        self.C13_lag = M * np.loadtxt(testpath + 'C_13_lag.dat')

        self.C1_sliding = M * np.loadtxt(testpath + 'C_1_sliding.dat')
        self.C7_sliding = M * np.loadtxt(testpath + 'C_7_sliding.dat')
        self.C13_sliding = M * np.loadtxt(testpath + 'C_13_sliding.dat')


    def tearDown(self):
        pass

    def test_count_matrix_bincount_mult(self):
        """Small test cases"""
        C = count_matrix_bincount_mult(self.dtrajs_short, 1, sliding=False).toarray()
        assert_allclose(C, self.B1_lag)

        C = count_matrix_bincount_mult(self.dtrajs_short, 2, sliding=False).toarray()
        assert_allclose(C, self.B2_lag)

        C = count_matrix_bincount_mult(self.dtrajs_short, 3, sliding=False).toarray()
        assert_allclose(C, self.B3_lag)

        C = count_matrix_bincount_mult(self.dtrajs_short, 1).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix_bincount_mult(self.dtrajs_short, 2).toarray()
        assert_allclose(C, self.B2_sliding)

        C = count_matrix_bincount_mult(self.dtrajs_short, 3).toarray()
        assert_allclose(C, self.B3_sliding)

        """Larger test cases"""
        C = count_matrix_bincount_mult(self.dtrajs_long, 1, sliding=False).toarray()
        assert_allclose(C, self.C1_lag)

        C = count_matrix_bincount_mult(self.dtrajs_long, 7, sliding=False).toarray()
        assert_allclose(C, self.C7_lag)

        C = count_matrix_bincount_mult(self.dtrajs_long, 13, sliding=False).toarray()
        assert_allclose(C, self.C13_lag)

        C = count_matrix_bincount_mult(self.dtrajs_long, 1).toarray()
        assert_allclose(C, self.C1_sliding)

        C = count_matrix_bincount_mult(self.dtrajs_long, 7).toarray()
        assert_allclose(C, self.C7_sliding)

        C = count_matrix_bincount_mult(self.dtrajs_long, 13).toarray()
        assert_allclose(C, self.C13_sliding)

        """Test raising of value error if lag greater than trajectory length"""
        with self.assertRaises(ValueError):
            C = count_matrix_bincount_mult(self.dtrajs_short, 10)

    def test_nstates_keyword(self):
        C = count_matrix_bincount_mult(self.dtrajs_short, 1, sliding=False, nstates=10).toarray()
        self.assertTrue(C.shape == (10, 10))


if __name__ == "__main__":
    unittest.main()