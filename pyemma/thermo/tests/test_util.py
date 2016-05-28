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
from pyemma.thermo.util.util import _ensure_umbrella_center
from pyemma.thermo.util.util import _ensure_force_constant

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
        us_center = _ensure_umbrella_center(1.0, 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3
        us_center = _ensure_umbrella_center(1.0, 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_tuple(self):
        # dimension=1, type=tuple
        us_center = _ensure_umbrella_center((1.0,), 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = _ensure_umbrella_center((1.0, 1.0, 1.0), 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))
        # dimension=4, not uniform
        us_center = _ensure_umbrella_center((1.0, 2.0, 3.0, 4.0), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = _ensure_umbrella_center(((1.0, 2.0, 3.0, 4.0),), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_list(self):
        # dimension=1
        us_center = _ensure_umbrella_center([1.0], 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = _ensure_umbrella_center([1.0, 1.0, 1.0], 3)
        self._assert_us_center(us_center, 3)
        # dimension=4, not uniform
        us_center = _ensure_umbrella_center([1.0, 2.0, 3.0, 4.0], 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = _ensure_umbrella_center([[1.0, 2.0, 3.0, 4.0],], 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    def test_ensure_umbrella_center_from_ndarray(self):
        # dimension=1
        us_center = _ensure_umbrella_center(np.array([1.0]), 1)
        self._assert_us_center(us_center, 1)
        np.testing.assert_array_equal(us_center, np.array([1.0], dtype=np.float64))
        # dimension=3, uniform
        us_center = _ensure_umbrella_center(np.array([1.0, 1.0, 1.0]), 3)
        self._assert_us_center(us_center, 3)
        np.testing.assert_array_equal(us_center, np.array([1.0, 1.0, 1.0], dtype=np.float64))
        # dimension=4, not uniform
        us_center = _ensure_umbrella_center(np.array([1.0, 2.0, 3.0, 4.0]), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        # dimension=4x1, not uniform
        us_center = _ensure_umbrella_center(np.array([[1.0, 2.0, 3.0, 4.0],]), 4)
        self._assert_us_center(us_center, 4)
        np.testing.assert_array_equal(us_center, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))


class TestProtectedUmbrellaSamplingForceMatrices(unittest.TestCase):

    def _assert_us_force_matrix(self, us_force_matrix, dimension):
        self.assertIsInstance(us_force_matrix, np.ndarray)
        self.assertTrue(us_force_matrix.dtype == np.float64)
        self.assertTrue(us_force_matrix.ndim == 2)
        self.assertTrue(us_force_matrix.shape[0] == dimension)
        self.assertTrue(us_force_matrix.shape[1] == dimension)

    def test_ensure_umbrella_force_matrix_from_scalar(self):
        # dimension=1
        us_force_matrix = _ensure_force_constant(1.0, 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2
        us_force_matrix = _ensure_force_constant(1.0, 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_tuple(self):
        # dimension=1
        us_force_matrix = _ensure_force_constant((1.0,), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = _ensure_force_constant(((1.0,),), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = _ensure_force_constant((1.0, 2.0), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = _ensure_force_constant(((1.0, 2.0), (3.0, 4.0)), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_list(self):
        # dimension=1
        us_force_matrix = _ensure_force_constant([1.0], 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = _ensure_force_constant([[1.0]], 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = _ensure_force_constant([1.0, 2.0], 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = _ensure_force_constant([[1.0, 2.0], [3.0, 4.0]], 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def test_ensure_umbrella_force_matrix_from_ndarray(self):
        # dimension=1
        us_force_matrix = _ensure_force_constant(np.array([1.0]), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=1x1
        us_force_matrix = _ensure_force_constant(np.array([[1.0]]), 1)
        self._assert_us_force_matrix(us_force_matrix, 1)
        np.testing.assert_array_equal(us_force_matrix, np.array([[1.0]], dtype=np.float64))
        # dimension=2, not uniform, diagonal
        us_force_matrix = _ensure_force_constant(np.array([1.0, 2.0]), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        # dimension=2, not uniform, not diagonal
        us_force_matrix = _ensure_force_constant(np.array([[1.0, 2.0], [3.0, 4.0]]), 2)
        self._assert_us_force_matrix(us_force_matrix, 2)
        np.testing.assert_array_equal(
            us_force_matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))












