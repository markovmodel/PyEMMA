# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader

class TestFragmentedTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #d = np.random.random((100, 3))
        d = np.array([[i] for i in range(0,100)])
        cls.d = d
        return cls

    def test_full_trajectory(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        reader.chunksize = 0
        expected = np.vstack((self.d, self.d))
        np.testing.assert_array_almost_equal(expected, reader.get_output(stride=1)[0])

    def test_full_trajectory_stridden(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        reader.chunksize = 0
        expected = np.vstack((self.d, self.d))[::3]
        out = reader.get_output(stride=3)[0]
        np.testing.assert_array_almost_equal(expected, out)

    def test_index_to_reader_index(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        assert (0,0) == reader._index_to_reader_index(0), "first frame is first frame of first reader"
        assert (0,1) == reader._index_to_reader_index(1), "second frame is second frame of first reader"
        assert (1,0) == reader._index_to_reader_index(100), "101'st frame is first frame of second reader"
        assert (1,1) == reader._index_to_reader_index(101), "102'nd frame is second frame of second reader"
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(-1)
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(200)

    def test_simple(self):
        pass
