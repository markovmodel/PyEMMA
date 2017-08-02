# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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


from __future__ import absolute_import
import unittest
import numpy as np

from pyemma.plots.plots2d import plot_free_energy, contour, scatter_contour


class TestPlots2d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.binomial(10, 0.4, (100, 2))

    def test_free_energy(self):
        plot_free_energy(self.data[:, 0], self.data[:, 1])

    def test_contour(self):
        contour(self.data[:,0], self.data[:,1], self.data[:,0])

    def test_scatter_contour(self):
        scatter_contour(self.data[:,0], self.data[:,1], self.data[:,0])
