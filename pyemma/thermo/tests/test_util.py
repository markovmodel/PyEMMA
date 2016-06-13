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
import pyemma.thermo.util.util as util

# ==================================================================================================
# tests for protected umbrella sampling convenience functions
# ==================================================================================================

class TestProtectedUmbrellaSamplingCenters(unittest.TestCase):

    def _assert_us_center(self, us_center, dimension):
        self.assertIsInstance(us_center, np.ndarray)
        self.assertTrue(us_center.dtype == np.float64)
        self.assertTrue(us_center.ndim == 1)
        self.assertTrue(us_center.shape[0] == dimension)

    def test_ensure_umbrella_center_from_scalar(self):
        # dimension=1
        us_center = util._ensure_umbrella_center(1.0, 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3
        us_center = util._ensure_umbrella_center(1.0, 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_tuple(self):
        # dimension=1, type=tuple
        us_center = util._ensure_umbrella_center((1.0,), 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = util._ensure_umbrella_center((1.0, 1.0, 1.0), 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))
        # dimension=4, not uniform
        us_center = util._ensure_umbrella_center((1.0, 2.0, 3.0, 4.0), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = util._ensure_umbrella_center(((1.0, 2.0, 3.0, 4.0),), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_list(self):
        # dimension=1
        us_center = util._ensure_umbrella_center([1.0], 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = util._ensure_umbrella_center([1.0, 1.0, 1.0], 3)
        self._assert_us_center(us_center, 3)
        # dimension=4, not uniform
        us_center = util._ensure_umbrella_center([1.0, 2.0, 3.0, 4.0], 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = util._ensure_umbrella_center([[1.0, 2.0, 3.0, 4.0],], 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_ndarray(self):
        # dimension=1
        us_center = util._ensure_umbrella_center(np.array([1.0]), 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = util._ensure_umbrella_center(np.array([1.0, 1.0, 1.0]), 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))
        # dimension=4, not uniform
        us_center = util._ensure_umbrella_center(np.array([1.0, 2.0, 3.0, 4.0]), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = util._ensure_umbrella_center(np.array([[1.0, 2.0, 3.0, 4.0],]), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    def test_ensure_umbrella_center_catches_unmatching_dimension(self):
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([1.0, 1.0], 1)
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([1.0, 1.0, 1.0], 2)
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([1.0, 1.0], 3)
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([[1.0], [1.0]], 1)
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([[1.0], [1.0]], 3)
        with self.assertRaises(ValueError):
            util._ensure_umbrella_center([[1.0, 1.0], [1.0]], 3)


class TestProtectedUmbrellaSamplingForceMatrices(unittest.TestCase):

    def _assert_us_force_matrix(self, us_force_matrix, dimension):
        self.assertIsInstance(us_force_matrix, np.ndarray)
        self.assertTrue(us_force_matrix.dtype == np.float64)
        self.assertTrue(us_force_matrix.ndim == 2)
        self.assertTrue(us_force_matrix.shape[0] == dimension)
        self.assertTrue(us_force_matrix.shape[1] == dimension)

    def test_ensure_umbrella_force_matrix_from_scalar(self):
        # dimension=1
        us_force_matrix = util._ensure_force_constant(1.0, 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2
        us_force_matrix = util._ensure_force_constant(1.0, 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_tuple(self):
        # dimension=1
        us_force_matrix = util._ensure_force_constant((1.0,), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = util._ensure_force_constant(((1.0,),), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = util._ensure_force_constant((1.0, 2.0), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = util._ensure_force_constant(((1.0, 2.0), (3.0, 4.0)), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_list(self):
        # dimension=1
        us_force_matrix = util._ensure_force_constant([1.0], 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = util._ensure_force_constant([[1.0]], 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = util._ensure_force_constant([1.0, 2.0], 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = util._ensure_force_constant([[1.0, 2.0], [3.0, 4.0]], 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_ndarray(self):
        # dimension=1
        us_force_matrix = util._ensure_force_constant(np.array([1.0]), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = util._ensure_force_constant(np.array([[1.0]]), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = util._ensure_force_constant(np.array([1.0, 2.0]), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = util._ensure_force_constant(np.array([[1.0, 2.0], [3.0, 4.0]]), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_catches_unmatching_dimension(self):
        with self.assertRaises(ValueError):
            util._ensure_force_constant([1.0, 1.0], 1)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([1.0, 1.0, 1.0], 2)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([1.0, 1.0], 3)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([[1.0], [1.0]], 1)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([[1.0], [1.0]], 3)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([[1.0, 1.0], 1.0], 3)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([1.0, [1.0, 1.0]], 3)
        with self.assertRaises(ValueError):
            util._ensure_force_constant([[1.0, 1.0], [1.0, 1.0]], 3)


class TestProtectedUmbrellaSamplingParameters(unittest.TestCase):

    def _assert_parameters(self,
        ttrajs, umbrella_centers, force_constants, unbiased_state,
        ref_ttrajs, ref_umbrella_centers, ref_force_constants, ref_unbiased_state):
        for ttraj, ref_ttraj in zip(ttrajs, ref_ttrajs):
            np.testing.assert_array_equal(ttraj, ref_ttraj)
        for center, ref_center in zip(umbrella_centers, ref_umbrella_centers):
            np.testing.assert_array_equal(center, ref_center)
        for force_constant, ref_force_constant in zip(force_constants, ref_force_constants):
            np.testing.assert_array_equal(force_constant, ref_force_constant)
        self.assertTrue(unbiased_state == ref_unbiased_state)

    def test_umbrella_sampling_parameters_1x0(self):
        ref_umbrella_centers = [0.0, 1.0]
        ref_force_constants = [1.0, 1.0]
        us_trajs = [np.array([0.0, 0.1, 0.2]), np.array([0.9, 1.0, 1.1])]
        # no md data
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1])],
            ref_umbrella_centers, ref_force_constants, None)
        # add md data
        md_trajs = [np.array([0.0, 0.5, 1.0])]
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
            ref_umbrella_centers, ref_force_constants, 2)
        # with kT parameter
        with self.assertRaises(ValueError):
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT=0.0)
        with self.assertRaises(ValueError):
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT='kT')
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT=2.0)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants * 2.0, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
            ref_umbrella_centers, ref_force_constants, 2)

    def test_umbrella_sampling_parameters_1x1(self):
        ref_umbrella_centers = [0.0, 1.0]
        ref_force_constants = [1.0, 1.0]
        us_trajs = [np.array([[0.0], [0.1], [0.2]]), np.array([[0.9], [1.0], [1.1]])]
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1])],
            ref_umbrella_centers, ref_force_constants, None)
        # add md data
        md_trajs = [np.array([[0.0], [0.5], [1.0]])]
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
            ref_umbrella_centers, ref_force_constants, 2)
        # with kT parameter
        with self.assertRaises(ValueError):
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT=0.0)
        with self.assertRaises(ValueError):
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT='kT')
        ttrajs, umbrella_centers, force_constants, unbiased_state = \
            util._get_umbrella_sampling_parameters(
                us_trajs, ref_umbrella_centers, ref_force_constants, md_trajs=md_trajs, kT=2.0)
        self._assert_parameters(
            ttrajs, umbrella_centers, force_constants * 2.0, unbiased_state,
            [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
            ref_umbrella_centers, ref_force_constants, 2)

    def test_umbrella_sampling_parameters_unmatching_dimensions(self):
        ref_umbrella_centers = [0.0, 1.0]
        ref_force_constants = [1.0, 1.0]
        us_trajs_x = [
            np.array([[0.0], [0.1], [0.2]]), np.array([[0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])]
        with self.assertRaises(ValueError):
            util._get_umbrella_sampling_parameters(
                us_trajs_x, ref_umbrella_centers, ref_force_constants)


class TestProtectedUmbrellaSamplingBiasSequence(unittest.TestCase):

    def _assert_bias_sequences(self, bias_sequences, ref_bias_sequences):
        self.assertTrue(len(bias_sequences) == len(ref_bias_sequences))
        for bs, rbs in zip(bias_sequences, ref_bias_sequences):
            np.testing.assert_array_equal(bs, rbs)

    def test_umbrella_sampling_bias_sequences_1x0(self):
        trajs = [np.array([0.0, 0.5, 1.0])]
        umbrella_centers = np.array([
            util._ensure_umbrella_center(0.0, 1),
            util._ensure_umbrella_center(1.0, 1)], dtype=np.float64)
        force_constants = np.array([
            util._ensure_force_constant(1.0, 1),
            util._ensure_force_constant(2.0, 1)], dtype=np.float64)
        self._assert_bias_sequences(
            util._get_umbrella_bias_sequences(trajs, umbrella_centers, force_constants),
            [np.array([[0.0, 1.0], [0.125, 0.25], [0.5, 0.0]])])

    def test_umbrella_sampling_bias_sequences_1x1(self):
        trajs = [np.array([[0.0], [0.5], [1.0]])]
        umbrella_centers = np.array([
            util._ensure_umbrella_center(0.0, 1),
            util._ensure_umbrella_center(1.0, 1)], dtype=np.float64)
        force_constants = np.array([
            util._ensure_force_constant(1.0, 1),
            util._ensure_force_constant(2.0, 1)], dtype=np.float64)
        self._assert_bias_sequences(
            util._get_umbrella_bias_sequences(trajs, umbrella_centers, force_constants),
            [np.array([[0.0, 1.0], [0.125, 0.25], [0.5, 0.0]])])

    def test_umbrella_sampling_bias_sequences_catches_unmatching_dimension(self):
        # wrong centers + constants
        with self.assertRaises(TypeError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0, 1.0]]), [[[1.0, 0.0], [1.0, 0.0]]])
        with self.assertRaises(TypeError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                [[1.0, 1.0]], np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([1.0, 1.0]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[[1.0, 1.0]]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0, 1.0]]), np.array([[[[1.0, 0.0], [1.0, 0.0]]]]))
        # conflicting centers + constants
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0, 1.0, 1.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]])],
                np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        # traj does not match valid centers + constants
        with self.assertRaises(TypeError):
            util._get_umbrella_bias_sequences([[[0.0, 0.0], [0.5, 0.1], [1.0, 0.2]]],
                np.array([[1.0, 1.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences([np.array([0.0, 0.5, 1.0])],
                np.array([[1.0, 1.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
        with self.assertRaises(ValueError):
            util._get_umbrella_bias_sequences(
                [np.array([[0.0, 1.0, 2.0], [0.5, 1.0, 2.0], [1.0, 1.0, 2.0]])],
                np.array([[1.0, 1.0]]), np.array([[[1.0, 0.0], [1.0, 0.0]]]))
