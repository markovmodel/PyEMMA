"""Unit tests for the count_matrix module"""

import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from os.path import abspath, join
from os import pardir

from pyemma.msm.estimation import count_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestCountMatrixMult(unittest.TestCase):
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
        C = count_matrix([self.S1, self.S2], 1, sliding=True).toarray()
        assert_allclose(C, self.B1_sliding)

        C = count_matrix([self.S1, self.S2], 2, sliding=True).toarray()
        assert_allclose(C, self.B2_sliding)

    def test_nstates_keyword(self):
        C = count_matrix([self.S1, self.S2], 1, sliding=True, nstates=10)
        self.assertTrue(C.shape == (10, 10))

        with self.assertRaises(ValueError):
            C = count_matrix([self.S1, self.S2], 1, sliding=True, nstates=1)


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


class TestArguments(unittest.TestCase):
    def testInputList(self):
        dtrajs = [0, 1, 2, 0, 0, 1, 2, 1, 0]
        count_matrix(dtrajs, 1)

    def testInput1Array(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        count_matrix(dtrajs, 1)

    def testInputNestedLists(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        count_matrix(dtrajs, 1)

    def testInputNestedListsDiffSize(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 2, 1]]
        count_matrix(dtrajs, 1)

    def testInputArray(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        count_matrix(dtrajs, 1)

    def testInputArrays(self):
        """ this is not supported, has to be list of ndarrays """
        dtrajs = np.array([[0, 1, 2, 0, 0, 1, 2, 1, 0],
                           [0, 1, 2, 0, 0, 1, 2, 1, 1]])

        with self.assertRaises(TypeError):
            count_matrix(dtrajs, 1)

    def testInputFloat(self):
        dtraj_with_floats = [0.0, 1, 0, 2, 3, 1, 0.0]
        # dtraj_int = [0, 1, 0, 2, 3, 1, 0]
        with self.assertRaises(TypeError):
            C_f = count_matrix(dtraj_with_floats, 1)
            # C_i = count_matrix(dtraj_int, 1)
            # np.testing.assert_array_equal(C_f.toarray(), C_i.toarray())


if __name__ == "__main__":
    unittest.main()

