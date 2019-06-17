# This file is part of PyEMMA.
#
# Copyright (c) 2017, 2018 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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


import unittest
import numpy as np
import matplotlib.pyplot as plt

from pyemma.plots.plots2d import contour, scatter_contour
from pyemma.plots.plots2d import plot_density
from pyemma.plots.plots2d import plot_free_energy
from pyemma.plots.plots2d import plot_contour
from pyemma.plots.plots2d import plot_state_map


class TestPlots2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.binomial(10, 0.4, (100, 2))

    def test_free_energy(self):
        fig, ax = plot_free_energy(
            self.data[:, 0], self.data[:, 1])
        plt.close(fig)

    def test_contour(self):
        ax = contour(self.data[:,0], self.data[:,1], self.data[:,0])
        plt.close(ax.get_figure())
        ax = contour(
            self.data[:,0], self.data[:,1], self.data[:,0],
            zlim=(self.data[:, 0].min(), self.data[:, 0].max()))
        plt.close(ax.get_figure())

    def test_scatter_contour(self):
        ax = scatter_contour(
            self.data[:,0], self.data[:,1], self.data[:,0])
        plt.close(ax.get_figure())

    def test_plot_density(self):
        fig, ax, misc = plot_density(
            self.data[:, 0], self.data[:, 1], logscale=True)
        plt.close(fig)
        fig, ax, misc = plot_density(
            self.data[:, 0], self.data[:, 1], logscale=False)
        plt.close(fig)
        fig, ax, misc = plot_density(
            self.data[:, 0], self.data[:, 1], alpha=True)
        plt.close(fig)
        fig, ax, misc = plot_density(
            self.data[:, 0], self.data[:, 1], zorder=-1)
        plt.close(fig)
        fig, ax, misc = plot_density(
            self.data[:, 0], self.data[:, 1],
            this_should_raise_a_UserWarning=True)
        plt.close(fig)

    def test_plot_free_energy(self):
        fig, ax, misc = plot_free_energy(
            self.data[:, 0], self.data[:, 1], legacy=False)
        plt.close(fig)
        with self.assertRaises(ValueError):
            plot_free_energy(
                self.data[:, 0], self.data[:, 1],
                legacy=False, offset=42)
        with self.assertRaises(ValueError):
            plot_free_energy(
                self.data[:, 0], self.data[:, 1],
                legacy=False, ncountours=42)

    def test_plot_contour(self):
        fig, ax, misc = plot_contour(
            self.data[:, 0], self.data[:, 1], self.data[:,0])
        plt.close(fig)
        fig, ax, misc = plot_contour(
            self.data[:, 0], self.data[:, 1], self.data[:,0],
            levels='legacy')
        plt.close(fig)
        fig, ax, misc = plot_contour(
            self.data[:, 0], self.data[:, 1], self.data[:,0],
            mask=True)
        plt.close(fig)

    def test_plot_state_map(self):
        fig, ax, misc = plot_state_map(
            self.data[:, 0], self.data[:, 1], self.data[:,0])
        plt.close(fig)
        fig, ax, misc = plot_state_map(
            self.data[:, 0], self.data[:, 1], self.data[:,0],
            zorder=0.5)
        plt.close(fig)
        fig, ax, misc = plot_state_map(
            self.data[:, 0], self.data[:, 1], self.data[:,0],
            cbar_orientation='horizontal', cbar_label=None)
        plt.close(fig)
        with self.assertRaises(ValueError):
            fig, ax, misc = plot_state_map(
                self.data[:, 0], self.data[:, 1], self.data[:,0],
                cbar_orientation='INVALID')
