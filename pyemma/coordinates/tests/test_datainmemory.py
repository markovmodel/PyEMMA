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


from __future__ import absolute_import
import pyemma
from six.moves import range
from six.moves import zip

'''
Created on 04.02.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.coordinates.data.data_in_memory import DataInMemory
from logging import getLogger

logger = getLogger('pyemma.'+'TestDataInMemory')


class TestDataInMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = np.random.random((100, 3))
        d_1d = np.random.random(100)

        cls.d = d
        cls.d_1d = d_1d
        return cls

    def test_skip(self):
        for skip in [0, 3, 13]:
            r1 = DataInMemory(self.d)
            out_with_skip = r1.get_output(skip=skip)[0]
            r2 = DataInMemory(self.d)
            out = r2.get_output()[0]
            np.testing.assert_almost_equal(out_with_skip, out[skip::],
                                           err_msg="The first %s rows were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def test_skip_input_list(self):
        for skip in [0, 3, 13]:
            r1 = DataInMemory([self.d, self.d])
            out_with_skip = r1.get_output(skip=skip)
            r2 = DataInMemory([self.d, self.d])
            out = r2.get_output()
            np.testing.assert_almost_equal(out_with_skip[0], out[0][skip::],
                                           err_msg="The first %s rows of the first file were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))
            np.testing.assert_almost_equal(out_with_skip[1], out[1][skip::],
                                           err_msg="The first %s rows of the second file were skipped, but that did not"
                                                   " match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def testWrongArguments(self):
        with self.assertRaises(ValueError):
            reader = DataInMemory("foo")

    def testListOfArrays(self):

        frames_per_traj = 100
        dim = 3
        data = [np.random.random((frames_per_traj, dim)) for _ in range(3)]

        d = DataInMemory(data)

        self.assertEqual(d.dimension(), dim)

        np.testing.assert_equal(
                d.trajectory_lengths(), np.array([frames_per_traj for _ in range(3)]))

    def testDataArray(self):
        frames_per_traj = 100
        dim = 3

        data = np.random.random((frames_per_traj, dim))
        d = DataInMemory(data)

        np.testing.assert_equal(
                d.trajectory_lengths(), np.array([frames_per_traj for _ in range(1)]))

    def test1dData(self):
        n = 3
        data = np.arange(n)
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), np.array([n]))
        self.assertEqual(reader.ndim, 1)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.n_frames_total(), n)

    def test1dDataList(self):
        n = 10
        data = [np.arange(n), np.arange(n)]
        reader = DataInMemory(data)

        np.testing.assert_equal(reader.trajectory_lengths(), np.array([n, n]))
        self.assertEqual(reader.ndim, 1)
        self.assertEqual(reader.number_of_trajectories(), 2)
        self.assertEqual(reader.n_frames_total(), 2 * n)

    def testNotEqualDims(self):
        """ should raise, since different dims can not be processed"""
        data = [np.zeros((10, 3)), np.zeros((10, 5))]

        with self.assertRaises(ValueError):
            DataInMemory(data)

    def test_ndim_input(self):
        data = np.empty((4, 2, 2, 2))

        reader = DataInMemory(data)

        self.assertEqual(reader.ndim, 2 * 2 * 2)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.n_frames_total(), 4)
        np.testing.assert_equal(
                reader.trajectory_lengths(), np.array([reader.n_frames_total()]))

    def test_time_lagged_chunked_access(self):
        n = 100
        data = [np.random.random((n, 3)), np.zeros((29, 3)),
                np.random.random((n - 50, 3))]
        reader = DataInMemory(data)
        self.assertEqual(reader.n_frames_total(), n + n - 50 + 29)

        # iterate over data
        it = reader.iterator(lag=30, return_trajindex=True)
        for itraj, X, Y in it:
            if itraj == 0:
                # self.assertEqual(X.shape, (100, 3)) <-- changed behavior: return only chunks of same size
                self.assertEqual(X.shape, (70, 3))
                self.assertEqual(Y.shape, (70, 3))
            elif itraj == 1:
                # the time lagged chunk can not be built due to lag time
                self.assertEqual(X.shape, (0, 3))
                self.assertEqual(Y.shape, (0, 3))
            elif itraj == 2:
                self.assertEqual(X.shape, (20, 3))
                self.assertEqual(Y.shape, (20, 3))

    def test_stride(self):
        reader = DataInMemory(self.d)
        stride = [1, 2, 3, 4, 5, 6, 7, 10, 11, 21, 23]
        for s in stride:
            output = reader.get_output(stride=s)[0]
            expected = self.d[::s]
            np.testing.assert_allclose(output, expected,
                                       err_msg="not equal for stride=%i" % s)

    def test_chunksize(self):
        data = np.random.randn(200, 2)
        cs = 100
        source = pyemma.coordinates.source(data, chunk_size=cs)
        source.chunksize = 100
        for i, ch in source.iterator():
            assert ch.shape[0] <= cs, ch.shape

    def test_lagged_iterator_1d(self):
        n = 57
        chunksize = 10
        lag = 1

        data = [np.arange(n), np.arange(50), np.arange(30)]
        input_lens = [x.shape[0] for x in data]
        reader = DataInMemory(data)
        reader.chunksize = chunksize

        self.assertEqual(reader.n_frames_total(), sum(input_lens))

        # store results by traj
        chunked_trajs = [[] for _ in range(len(data))]
        chunked_lagged_trajs = [[] for _ in range(len(data))]

        # iterate over data
        for itraj, X, Y in reader.iterator(lag=lag):
            chunked_trajs[itraj].append(X)
            chunked_lagged_trajs[itraj].append(Y)

        trajs = [np.vstack(ichunks) for ichunks in chunked_trajs]
        lagged_trajs = [np.vstack(ichunks) for ichunks in chunked_lagged_trajs]

        # unlagged data
        for traj, input_traj in zip(trajs, data):
            # do not consider chunks that have no lagged counterpart
            input_shape = input_traj.shape
            np.testing.assert_equal(traj.reshape((input_shape[0] - lag,)), input_traj[:len(input_traj) - lag])

        # lagged data
        lagged_0 = [d[lag:] for d in data]

        for traj, input_traj in zip(lagged_trajs, lagged_0):
            np.testing.assert_equal(traj.reshape(input_traj.shape), input_traj)

    def test_lagged_iterator_2d(self):
        chunksize = 10
        lag = 1

        data = [np.arange(300).reshape((100, 3)),
                np.arange(29 * 3).reshape((29, 3)),
                np.arange(150).reshape(50, 3)]
        input_lens = [x.shape[0] for x in data]
        # print data[0].shape
        reader = DataInMemory(data)
        reader.chunksize = chunksize

        self.assertEqual(reader.n_frames_total(), sum(input_lens))

        # store results by traj
        chunks = [[] for _ in range(len(data))]
        lagged_chunks = [[] for _ in range(len(data))]

        # iterate over data
        for itraj, X, Y in reader.iterator(lag=lag):
            chunks[itraj].append(X)
            lagged_chunks[itraj].append(Y)

        trajs = [np.vstack(ichunks) for ichunks in chunks]

        lagged_trajs = [np.vstack(ichunks) for ichunks in lagged_chunks]

        # unlagged data
        for traj, input_traj in zip(trajs, data):
            # do not consider chunks that have no lagged counterpart
            input_shape = input_traj.shape
            np.testing.assert_equal(traj.reshape((input_shape[0] - lag, 3)), input_traj[:len(input_traj) - lag])

        # lagged data
        lagged_0 = [d[lag:] for d in data]

        for traj, input_traj in zip(lagged_trajs, lagged_0):
            np.testing.assert_equal(traj.reshape(input_traj.shape), input_traj)

    def test_lagged_stridden_access(self):
        data = np.random.random((1000, 2))
        reader = DataInMemory(data)
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        for stride in strides:
            for lag in lags:
                chunks = []
                for _, _, Y in reader.iterator(stride=stride, lag=lag):
                    chunks.append(Y)
                chunks = np.vstack(chunks)
                np.testing.assert_equal(chunks, data[lag::stride], "failed for stride=%s, lag=%s" % (stride, lag))

    def test_cols(self):
        reader = DataInMemory(self.d)
        cols=(2, 0)
        for x in reader.iterator(chunk=0, return_trajindex=False, cols=cols):
            np.testing.assert_equal(x, self.d[:, cols])

if __name__ == "__main__":
    unittest.main()
