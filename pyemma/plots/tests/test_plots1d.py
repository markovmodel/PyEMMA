# This file is part of PyEMMA.
#
# Copyright (c) 2018 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pyemma.plots.plots1d import plot_feature_histograms


class TestPlots1d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.rand(500, 10)

    def test_feature_histograms(self):
        plot_feature_histograms(self.data)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_feature_histograms(self.data, feature_labels=np.random.rand(5))

    def test_feature_histograms_mpl_arguments(self):
        labels = ['PyEMMA' for _ in range(self.data.shape[1])]
        plot_feature_histograms(self.data,
                                feature_labels=labels,
                                ylog=True,
                                n_bins=10,
                                color='g')

    def test_feature_histograms_ax_argument(self):
        from matplotlib.pyplot import subplots
        fig, ax = subplots()
        plot_feature_histograms(self.data, ax=ax)
