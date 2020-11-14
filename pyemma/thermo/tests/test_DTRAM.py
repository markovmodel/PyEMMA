# This file is part of PyEMMA.
#
# Copyright (c) 2020 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import pyemma.thermo


class TestDTRAMConnectedSet(unittest.TestCase):
    def test_DTRAM_connected_set(self):
        dtrajs = [np.array([3, 0, 1, 0, 1]), np.array([3, 1, 2, 1, 2])]
        ttrajs = [np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1])]
        dtram = pyemma.thermo.DTRAM(bias_energies_full=np.zeros((2, 4)), lag=1)
        dtram.estimate((ttrajs, dtrajs))
        np.testing.assert_allclose(dtram.active_set, [0, 1, 2])
        np.testing.assert_allclose(dtram.models[0].active_set, [0, 1])
        np.testing.assert_allclose(dtram.models[1].active_set, [1, 2])
        np.testing.assert_allclose(dtram.stationary_distribution, np.ones(3)/3)
        np.testing.assert_allclose(dtram.stationary_distribution_full_state, np.array([1., 1., 1., 0.])/3)
        np.testing.assert_allclose(dtram.models[0].stationary_distribution, np.array([0.5, 0.5]))
        np.testing.assert_allclose(dtram.models[0].stationary_distribution_full_state, np.array([0.5, 0.5, 0., 0.]))
        np.testing.assert_allclose(dtram.models[1].stationary_distribution, np.array([0.5, 0.5]))
        np.testing.assert_allclose(dtram.models[1].stationary_distribution_full_state, np.array([0., 0.5, 0.5, 0.]))


if __name__ == "__main__":
    unittest.main()
