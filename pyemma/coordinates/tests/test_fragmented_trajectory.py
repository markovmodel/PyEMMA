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
        d = np.array([[i] for i in range(0, 100)])
        cls.d = d
        return cls

    def test_full_trajectory(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        reader.chunksize = 0
        expected = np.vstack((self.d, self.d))
        np.testing.assert_array_almost_equal(expected, reader.get_output(stride=1)[0])

    def test_full_trajectory_stridden(self):
        for stride in [1, 3, 5, 7, 13, 20]:
            reader = FragmentedTrajectoryReader([self.d, self.d])
            reader.chunksize = 0
            expected = np.vstack((self.d, self.d))[::stride]
            out = reader.get_output(stride=stride)[0]
            np.testing.assert_array_almost_equal(expected, out)

    def test_full_trajectory_stridden_with_lag(self):
        data = np.vstack((self.d, self.d))
        for lag in [1, 5, 7]:
            for stride in [1, 3, 5, 7, 13, 20]:
                reader = FragmentedTrajectoryReader([self.d, self.d])
                reader.chunksize = 0

                X, Y = None, None
                # not chunked
                for itraj, X, Y in reader.iterator(stride=stride, lag=lag):
                    pass

                np.testing.assert_array_almost_equal(data[::stride], X)
                np.testing.assert_array_almost_equal(data[lag::stride], Y)

    def test_chunked_trajectory(self):
        data = np.vstack((self.d, self.d))
        lag = 1
        stride = 1
        chunksize = 1
        # for lag in [0, 1, 3]:
        #     for stride in [1, 3, 5]:
        #         for chunksize in [1, 17]:
        #             print "lag=%s, stride=%s, cs=%s" % (lag, stride, chunksize)
        reader = FragmentedTrajectoryReader([self.d, self.d])
        reader.chunksize = chunksize

        if lag > 0:
            collected = None
            collected_lagged = None
            for itraj, X, Y in reader.iterator(stride=stride, lag=lag):
                collected = X if collected is None else np.vstack((collected, X))
                collected_lagged = Y if collected_lagged is None else np.vstack((collected_lagged, X))
            np.testing.assert_array_almost_equal(data[::stride], collected)
            np.testing.assert_array_almost_equal(data[lag::stride], collected)
        else:
            collected = None
            for itraj, X in reader.iterator(stride=stride):
                collected = X if collected is None else np.vstack((collected, X))
            np.testing.assert_array_almost_equal(data[::stride], collected)




    def test_index_to_reader_index(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        assert (0, 0) == reader._index_to_reader_index(0), "first frame is first frame of first reader"
        assert (0, 1) == reader._index_to_reader_index(1), "second frame is second frame of first reader"
        assert (1, 0) == reader._index_to_reader_index(100), "101'st frame is first frame of second reader"
        assert (1, 1) == reader._index_to_reader_index(101), "102'nd frame is second frame of second reader"
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(-1)
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(200)

    def test_simple(self):
        pass
