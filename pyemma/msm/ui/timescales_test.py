__author__ = 'noe'

import unittest
import numpy as np

from timescales import ImpliedTimescales

class ImpliedTimescalesTest(unittest.TestCase):

    def setUp(self):
        self.dtrajs = []

        # simple case
        dtraj_simple = [0,1,1,1,0]
        self.dtrajs.append([dtraj_simple])

        # as ndarray
        self.dtrajs.append([np.array(dtraj_simple)])

        dtraj_disc = [0,1,1,0,0]
        self.dtrajs.append([dtraj_disc])

        # multitrajectory case
        self.dtrajs.append([[0],[1,1,1,1],[0,1,1,1,0],[0,1,0,1,0,1,0,1]])


    def compute_nice(self, reversible):
        """
        Tests if standard its estimates run without errors

        :return:
        """
        for i in range(len(self.dtrajs)):
            its = ImpliedTimescales(self.dtrajs[i], reversible=reversible)
            print its.get_lagtimes()
            print its.get_timescales()


    def test_nice_sliding_rev(self):
        """
        Tests if nonreversible sliding estimate runs without errors
        :return:
        """
        self.compute_nice(True)

    def test_nice_sliding_nonrev(self):
        """
        Tests if nonreversible sliding estimate runs without errors
        :return:
        """
        self.compute_nice(False)


if __name__ == "__main__":
    unittest.main()
