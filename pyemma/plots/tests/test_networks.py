
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from msmtools.analysis import mfpt
import msmtools
import matplotlib
import matplotlib.pyplot as plt


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

        cls.P_mfpt = np.zeros_like(cls.P)
        for ii in np.arange(cls.P.shape[0]):
            for jj in np.arange(cls.P.shape[1]):
                cls.P_mfpt[ii,jj] = mfpt(cls.P, [ii], [jj])
        return cls

    def test_flux(self):
        r = tpt(self.P, self.A, self.B)
        fig, pos = plot_flux(r)
        assert type(fig) is matplotlib.figure.Figure
        # matplotlib.pyplot.show(fig)
        # x values should be close to committor
        np.testing.assert_allclose(pos[:,0], r.committor)

    def test_random(self):
        C = np.random.randint(0, 1000, size=(10, 10))
        P = msmtools.estimation.transition_matrix(C, reversible=True)
        r = tpt(P, [0], [len(C)-1])
        fig, pos = plot_flux(r)
        assert type(fig) is matplotlib.figure.Figure

    def test_plot_markov_model(self):
        fig, pos = plot_markov_model(self.P)
        assert type(fig) is matplotlib.figure.Figure

    def test_forced_no_arrows_labels(self):
        fig, pos = plot_markov_model(self.P, arrow_labels=None)
        assert type(fig) is matplotlib.figure.Figure

    def test_numeric_arrow_labels(self):
        fig, pos = plot_markov_model(self.P, arrow_labels=self.P)
        assert type(fig) is matplotlib.figure.Figure

    def test_alphanumeric_arrow_labels(self):
        fig, pos = plot_markov_model(
            self.P, arrow_labels=self.P_mfpt, arrow_label_format='mfpts = %f frames')
        assert type(fig) is matplotlib.figure.Figure

    def test_alphanumeric_arrow_labels_using_ax(self):
        orig_fig, ax = plt.subplots()
        fig, pos = plot_markov_model(
            self.P, arrow_labels=self.P_mfpt, arrow_label_format='mfpts = %f frames', ax=ax)
        assert type(fig) is matplotlib.figure.Figure
        assert fig == orig_fig

if __name__ == "__main__":
    unittest.main()
