'''
Created on 22.05.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.plots.networks import plot_flux
from pyemma.msm.flux.api import tpt


class TestNetworkPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 5-state toy system
        cls.P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                          [0.1, 0.75, 0.05, 0.05, 0.05],
                          [0.05, 0.1, 0.8, 0.0, 0.05],
                          [0.0, 0.2, 0.0, 0.8, 0.0],
                          [0.0, 0.02, 0.02, 0.0, 0.96]])
        cls.A = [0]
        cls.B = [4]
        return cls

    def test_flux(self):
        r = tpt(self.P, self.A, self.B)
        plot_flux(r)


if __name__ == "__main__":
    unittest.main()
