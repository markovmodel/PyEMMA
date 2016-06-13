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
from pyemma.thermo import estimate_umbrella_sampling
from pyemma.coordinates import cluster_regspace

# ==================================================================================================
# tests for the umbrella sampling API
# ==================================================================================================

class TestProtectedUmbrellaSamplingCenters(unittest.TestCase):

    def test_exceptions(self):
        us_centers = [1.1, 1.3]
        us_force_constants = [1.0, 1.0]
        us_trajs = [
            np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.1]),
            np.array([1.3, 1.2, 1.3, 1.4, 1.4, 1.3])]
        md_trajs = [
            np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
            np.array([1.5, 1.4, 1.3, 1.4, 1.4, 1.5])]
        cluster = cluster_regspace(data=us_trajs+md_trajs, max_centers=10, dmin=0.15)
        us_dtrajs = cluster.dtrajs[:2]
        md_dtrajs = cluster.dtrajs[2:]
        # unmatching number of us trajectories / us parameters
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs[:-1], us_dtrajs, us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs[:-1], us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers[:-1], us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants[:-1])
        # unmatching number of md trajectories
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs[:-1], md_dtrajs=md_dtrajs)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs, md_dtrajs=md_dtrajs[:-1])
        # unmatchig trajectory lengths
        us_trajs_x = [
            np.array([1.0, 1.1, 1.2, 1.1, 1.0]),
            np.array([1.3, 1.2, 1.3, 1.4, 1.4])]
        md_trajs_x = [
            np.array([0.9, 1.0, 1.1, 1.2, 1.3]),
            np.array([1.5, 1.4, 1.3, 1.4, 1.4])]
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs_x, us_dtrajs, us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs_x, md_dtrajs=md_dtrajs)
        # unmatching md_trajs/md_dtrajs cases
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=None, md_dtrajs=md_dtrajs)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs, md_dtrajs=None)
        # single trajectory cases
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs[0], us_dtrajs[0], us_centers[0], us_force_constants[0])
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs[0], md_dtrajs=md_dtrajs[0])

