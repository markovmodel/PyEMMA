
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on 22.05.2015

@author: marscher
'''

from __future__ import absolute_import
import unittest
import numpy as np

from pyemma.plots.networks import plot_flux, plot_markov_model
#from pyemma.msm.flux.api import tpt
from msmtools.flux import tpt
import msmtools
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
        P = msmtools.estimation.transition_matrix(C, reversible=True)
        r = tpt(P, [0], [len(C)-1])
        fig, pos = plot_flux(r)

    def test_plot_markov_model(self):
        fig, pos = plot_markov_model(self.P)
        assert type(fig) is matplotlib.figure.Figure

if __name__ == "__main__":
    unittest.main()