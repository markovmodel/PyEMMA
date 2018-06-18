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
import warnings

from pyemma.plots.plots1d import plot_feature_histograms


class TestPlots1d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.rand(500, 10)

    def test_feature_histograms(self):
        plot_feature_histograms(self.data)

    def test_feature_histograms_nowarning(self):
        with warnings.catch_warnings(record=True) as w:
            plot_feature_histograms(self.data)
            assert len(w) == 0

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            plot_feature_histograms(self.data, feature_labels=np.random.rand(5))
