
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

from pyemma.plots.networks import plot_flux, plot_markov_model, plot_network
from pyemma.msm import tpt, MSM
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
        cls.msm = MSM(cls.P)

        cls.P_mfpt = np.zeros_like(cls.P)
        for ii in np.arange(cls.P.shape[0]):
            for jj in np.arange(cls.P.shape[1]):
                cls.P_mfpt[ii,jj] = cls.msm.mfpt([ii], [jj])
        return cls

    def test_flux(self):
        r = tpt(self.msm, self.A, self.B)
        fig, pos = plot_flux(r)
        assert type(fig) is matplotlib.figure.Figure
        # matplotlib.pyplot.show(fig)
        # x values should be close to committor
        np.testing.assert_allclose(pos[:,0], r.committor)

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

    def test_string_arrow_labels_using_ax(self):
        orig_fig, ax = plt.subplots()
        labels = np.array([['A2A', 'A2B'],
                           ['B2A', 'B2B']]
                  )
        fig, pos = plot_network(
            self.P[:2,:2], arrow_labels=labels, ax=ax)
        assert type(fig) is matplotlib.figure.Figure

    def test_state_labels_network(self):
        state_labels = [str(i) for i in np.arange(len(self.P))]
        plot_network(self.P, state_labels=state_labels)

        with self.assertRaises(ValueError):
            plot_network(self.P, state_labels=state_labels[:2])

        state_labels = 'auto'
        plot_network(self.P, state_labels=state_labels)

    def test_state_labels_flux(self):
        """ ensure our labels show up in the plot"""
        flux = tpt(self.msm, [0,1], [2,4])
        labels = ['foo', '0', '1', '2', 'bar']
        fig, pos = plot_flux(flux, state_labels=labels)
        labels_in_fig = np.array([text.get_text() for text in fig.axes[0].texts])

        for l in labels:
            self.assertEqual((labels_in_fig == l).sum(), 1)

    def test_state_labels_flux_auto(self):
        """ ensure auto generated labels show up in the plot"""
        A = [0,1]
        B = [2,4]
        flux = tpt(self.msm, A, B)
        fig, pos = plot_flux(flux, state_labels='auto')
        labels_in_fig = np.array([text.get_text() for text in fig.axes[0].texts])
        self.assertEqual((labels_in_fig == "A").sum(), len(A))
        self.assertEqual((labels_in_fig == "B").sum(), len(B))

if __name__ == "__main__":
    unittest.main()
