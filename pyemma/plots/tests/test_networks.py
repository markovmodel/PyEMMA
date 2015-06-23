'''
Created on 22.05.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.plots.networks import plot_flux, plot_markov_model
from pyemma.msm.flux.api import tpt
import pyemma
import matplotlib


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
        fig, pos = plot_flux(r)
        assert type(fig) is matplotlib.figure.Figure
#        matplotlib.pyplot.show(fig)
        # x values should be close to committor
        np.testing.assert_allclose(pos[:,0], r.committor)

    def test_random(self):
        C = np.random.randint(0, 1000, size=(10, 10))
        P = pyemma.msm.estimation.transition_matrix(C, reversible=True)
        r = tpt(P, [0], [len(C)-1])
        fig, pos = plot_flux(r)

    def test_plot_markov_model(self):
        from pyemma.msm.analysis import stationary_distribution
        fig, pos = plot_markov_model(self.P)
        assert type(fig) is matplotlib.figure.Figure

        #matplotlib.pyplot.show(fig)

if __name__ == "__main__":
    unittest.main()
