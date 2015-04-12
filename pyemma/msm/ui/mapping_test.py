"""Unit test for the reaction pathway decomposition

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest
import numpy as np

from pyemma.util.numeric import assert_allclose

from mapping import from_lcc_labels, to_lcc_labels


class TestLccLabels(unittest.TestCase):

    def setUp(self):
        self.lcc = np.array([1, 3, 5])
        self.ix = np.arange(self.lcc.shape[0])

        self.states_in_lcc = np.array([0, 2, 0, 0, 1])
        self.states = np.array([1, 5, 1, 1, 3])

    def test_from_lcc_labels(self):
        states_in_lcc = to_lcc_labels(self.states, self.lcc)
        assert_allclose(states_in_lcc, self.states_in_lcc)

    def test_to_lcc_labels(self):
        states = from_lcc_labels(self.states_in_lcc, self.lcc)
        assert_allclose(states, self.states)

if __name__ == "__main__":
    unittest.main()
