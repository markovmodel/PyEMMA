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
import os
import unittest

import pkg_resources
import mdtraj
import numpy as np
import pyemma.coordinates as coor
from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
from six.moves import range


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

    def test_full_trajectory_random_access(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        indices = np.asarray([[0, 1], [0, 3], [0, 3], [0, 99], [0, 100], [0, 199]])
        out = reader.get_output(stride=indices, chunk=0)
        np.testing.assert_array_equal(np.array(out).squeeze(), np.array([1, 3, 3, 99, 0, 99]))

    def test_chunked_trajectory_random_access(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        indices = np.asarray([[0, 1], [0, 3], [0, 3], [0, 99], [0, 100], [0, 199]])
        out = reader.get_output(stride=indices, chunk=1)
        np.testing.assert_array_equal(np.array(out).squeeze(), np.array([1,3,3,99,0,99]))

    def test_full_trajectory_stridden(self):
        for stride in [1, 3, 5, 7, 13, 20]:
            reader = FragmentedTrajectoryReader([self.d, self.d])
            reader.chunksize = 0
            expected = np.vstack((self.d, self.d))[::stride]
            out = reader.get_output(stride=stride)[0]
            np.testing.assert_array_almost_equal(expected, out, err_msg="Failed for stride=%s" % stride)

    def test_full_trajectory_stridden_with_lag(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        data = np.vstack((self.d, self.d))
        for lag in [1, 5, 7]:
            for stride in [1, 3, 5, 7, 13, 20]:
                reader.chunksize = 0

                X, Y = None, None
                # not chunked
                for itraj, X, Y in reader.iterator(stride=stride, lag=lag):
                    pass

                np.testing.assert_array_almost_equal(data[::stride][0:len(Y)], X)
                np.testing.assert_array_almost_equal(data[lag::stride], Y)

    def test_fragmented_xtc(self):
        from pyemma.coordinates.tests.util import create_traj

        top_file = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        trajfiles = []
        for _ in range(3):
            f, _, _ = create_traj(top_file)
            trajfiles.append(f)
        try:
            # three trajectories: one consisting of all three, one consisting of the first,
            # one consisting of the first and the last
            source = coor.source([trajfiles, [trajfiles[0]], [trajfiles[0], trajfiles[2]]], top=top_file)
            source.chunksize = 1000

            out = source.get_output(stride=1)
            trajs = [mdtraj.load(trajfiles[i], top=top_file).xyz.reshape(-1,9) for i in range(0,3)]

            np.testing.assert_equal(out[0], np.vstack(trajs))
            np.testing.assert_equal(out[1], trajs[0])
            np.testing.assert_equal(out[2], np.vstack((trajs[0], trajs[2])))
        finally:
            for t in trajfiles:
                try:
                    os.unlink(t)
                except EnvironmentError:
                    pass

    def test_multiple_input_trajectories_random_access(self):
        indices = np.asarray([
            [0, 1], [0, 3], [0, 3], [0, 99], [0, 100], [0, 199],
            [1, 0], [1, 5], [1, 99],
            [2, 5], [2, 7], [2, 23]
        ])
        expected = [np.array([1, 3, 3, 99, 0, 99]), np.array([0, 5, 99]), np.array([5, 7, 23])]
        for chunk_size in [0, 1, 3, 5, 13]:
            reader = FragmentedTrajectoryReader([[self.d, self.d], self.d, [self.d, self.d]])
            out_full_trajectory_mode = reader.get_output(chunk=chunk_size, stride=indices)
            for i in range(3):
                np.testing.assert_array_equal(expected[i], out_full_trajectory_mode[i].squeeze())

    def test_multiple_input_trajectories(self):
        reader = FragmentedTrajectoryReader([[self.d, self.d], self.d, [self.d, self.d]])
        reader.chunksize = 37
        out = reader.get_output()
        reader.chunksize = 0
        out2 = reader.get_output()
        expected0_2 = np.vstack((self.d, self.d))
        for itraj in range(0, 3):
            np.testing.assert_array_almost_equal(out[itraj], out2[itraj])
        np.testing.assert_array_almost_equal(out[0], expected0_2)
        np.testing.assert_array_almost_equal(out[1], self.d)
        np.testing.assert_array_almost_equal(out[2], expected0_2)

    def test_chunked_trajectory_with_lag(self):
        data = np.vstack((self.d, self.d))
        reader = FragmentedTrajectoryReader([self.d, self.d])
        for lag in [0, 1, 3]:
            for stride in [1, 3, 5]:
                for chunksize in [1, 34, 53, 72]:
                    reader.chunksize = chunksize
                    if lag > 0:
                        collected = None
                        collected_lagged = None
                        for itraj, X, Y in reader.iterator(stride=stride, lag=lag):
                            collected = X if collected is None else np.vstack((collected, X))
                            collected_lagged = Y if collected_lagged is None else np.vstack((collected_lagged, Y))
                        np.testing.assert_array_almost_equal(data[::stride][0:len(collected_lagged)], collected)
                        np.testing.assert_array_almost_equal(data[lag::stride], collected_lagged)
                    else:
                        collected = None
                        for itraj, X in reader.iterator(stride=stride):
                            collected = X if collected is None else np.vstack((collected, X))
                        np.testing.assert_array_almost_equal(data[::stride], collected)

    def test_index_to_reader_index(self):
        reader = FragmentedTrajectoryReader([self.d, self.d])
        assert (0, 0) == reader._index_to_reader_index(0, 0), "first frame is first frame of first reader"
        assert (0, 1) == reader._index_to_reader_index(1, 0), "second frame is second frame of first reader"
        assert (1, 0) == reader._index_to_reader_index(100, 0), "101'st frame is first frame of second reader"
        assert (1, 1) == reader._index_to_reader_index(101, 0), "102'nd frame is second frame of second reader"
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(-1, 0)
        with self.assertRaises(ValueError):
            reader._index_to_reader_index(200, 0)

    def test_cols(self):
        dim = 5
        arr = np.arange(60).reshape(-1, dim)
        data = [(arr, arr), arr, (arr, arr, arr)]
        reader = FragmentedTrajectoryReader(data)
        cols = (0, 3)
        for itraj, x in reader.iterator(chunk=0, return_trajindex=True, cols=cols):
            if isinstance(data[itraj], tuple):
                syn_traj = np.concatenate(data[itraj])
            else:
                syn_traj = data[itraj]
            np.testing.assert_equal(x, syn_traj[:, cols])

    def test_raise_different_dims(self):
        data = [self.d, np.array([[1,2,3], [4,5,6]])]
        with self.assertRaises(ValueError):
            FragmentedTrajectoryReader(data)
