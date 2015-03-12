"""Unit tests for the count_matrix module"""

import unittest

import numpy as np

from os.path import abspath, join
from os import pardir

import pyemma.msm.estimation as msmest

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'

class TestCount(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_count_1(self):
        S = np.array([0, 0, 0, 0, 0, 0])
        H = np.array([6])
        assert(msmest.number_of_states(S) == 1)
        assert(msmest.number_of_states(S, only_used=True) == 1)
        assert(np.allclose(msmest.count_states(S),H))

    def test_count_2(self):
        S = np.array([1, 1, 1, 1, 1, 1])
        H = np.array([0,6])
        assert(msmest.number_of_states(S) == 2)
        assert(msmest.number_of_states(S, only_used=True) == 1)
        assert(np.allclose(msmest.count_states(S),H))

    def test_count_3(self):
        S1 = np.array([0, 1, 2, 3, 4])
        S2 = np.array([2, 2, 2, 2, 6])
        H = np.array([1, 1, 5, 1, 1, 0, 1])
        assert(msmest.number_of_states([S1,S2]) == 7)
        assert(msmest.number_of_states([S1,S2], only_used=True) == 6)
        assert(np.allclose(msmest.count_states([S1,S2]),H))


if __name__=="__main__":
    unittest.main()

