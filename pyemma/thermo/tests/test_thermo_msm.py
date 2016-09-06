# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import numpy.testing as npt
from pyemma.thermo.models.memm import ThermoMSM
from pyemma.msm import MSM


class TestThermoMSM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nstates_full = 10 + np.random.randint(10)
        cls.P_full = np.eye(cls.nstates_full)
        cls.nstates = 6 + np.random.randint(4)
        cls.active_set = np.sort(np.random.permutation(np.arange(cls.nstates_full))[:cls.nstates])
        cls.P = 1.0E-5 + np.random.rand(cls.nstates, cls.nstates)
        cls.P += cls.P.T
        cls.P /= cls.P.sum()
        cls.pi = cls.P.sum(axis=1)
        cls.P /= cls.pi[:, np.newaxis]
        for i, k in enumerate(cls.active_set):
            cls.P_full[k, cls.active_set] = cls.P[i, :]
        cls.pi_full = np.zeros(cls.nstates_full)
        cls.pi_full[cls.active_set] = cls.pi
        cls.f = -np.log(cls.pi)
        cls.f_full = np.inf * np.ones(cls.nstates_full)
        cls.f_full[cls.active_set] = cls.f
        msm = MSM(cls.P)
        cls.eigvec_l_full = np.inf * np.ones(shape=(cls.nstates-1, cls.nstates_full))
        cls.eigvec_r_full = np.inf * np.ones(shape=(cls.nstates_full, cls.nstates-1))
        cls.eigvec_l_full[:, cls.active_set] = msm.eigenvectors_left(k=cls.nstates-1)
        cls.eigvec_r_full[cls.active_set, :] = msm.eigenvectors_right(k=cls.nstates-1)
        cls.msm = ThermoMSM(cls.P, cls.active_set, cls.nstates_full, cls.pi)

    def test_f(self):
        npt.assert_array_equal(self.msm.free_energies, self.f)
        npt.assert_array_equal(self.msm.free_energies_full_state, self.f_full)
        npt.assert_array_equal(self.msm.f, self.f)
        npt.assert_array_equal(self.msm.f_full_state, self.f_full)

    def test_pi(self):
        npt.assert_array_equal(self.msm.pi, self.pi)
        npt.assert_array_equal(self.msm.pi_full_state, self.pi_full)
        npt.assert_array_equal(self.msm.stationary_distribution, self.pi)
        npt.assert_array_equal(self.msm.stationary_distribution_full_state, self.pi_full)

    def test_eigenvectors(self):
        npt.assert_array_equal(
            self.msm.eigenvectors_left_full_state(k=self.nstates-1), self.eigvec_l_full)
        npt.assert_array_equal(
            self.msm.eigenvectors_right_full_state(k=self.nstates-1), self.eigvec_r_full)
