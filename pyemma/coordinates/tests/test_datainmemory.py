# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import pyemma

'''
Created on 04.02.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.util.log import getLogger

logger = getLogger('TestDataInMemory')


class TestDataInMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = np.random.random((100, 3))
        d_1d = np.random.random(100)

        cls.d = d
        cls.d_1d = d_1d
        return cls

    def testWrongArguments(self):
        with self.assertRaises(ValueError):
            reader = DataInMemory("foo")

    def testListOfArrays(self):

        frames_per_traj = 100
        dim = 3
        data = [np.random.random((frames_per_traj, dim)) for _ in xrange(3)]

        d = DataInMemory(data)

        self.assertEqual(d.dimension(), dim)

        np.testing.assert_equal(
            d.trajectory_lengths(), np.array([frames_per_traj for _ in xrange(3)]))

    def testDataArray(self):
        frames_per_traj = 100
        dim = 3

        data = np.random.random((frames_per_traj, dim))
        d = DataInMemory(data)

        np.testing.assert_equal(
            d.trajectory_lengths(), np.array([frames_per_traj for _ in xrange(1)]))

    def test1dData(self):
        n = 3
        data = np.arange(n)
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), np.array([n]))
        self.assertEqual(reader.dimension(), 1)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.n_frames_total(), n)

    def test1dDataList(self):
        n = 10
        data = [np.arange(n), np.arange(n)]
        reader = DataInMemory(data)

        np.testing.assert_equal(reader.trajectory_lengths(), np.array([n, n]))
        self.assertEqual(reader.dimension(), 1)
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

        self.assertEqual(reader.dimension(), 2 * 2 * 2)
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
        lag = 30
        t = 0
        itraj = 0
        last_chunk = False
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                X, Y = reader._next_chunk(lag=lag)
                if itraj == 0:
                    self.assertEqual(X.shape, (100, 3))
                    self.assertEqual(Y.shape, (70, 3))
                elif itraj == 1:
                    # the time lagged chunk can not be built due to lag time
                    self.assertEqual(X.shape, (29, 3))
                    self.assertEqual(Y.shape, (0, 3))
                elif itraj == 2:
                    self.assertEqual(X.shape, (50, 3))
                    self.assertEqual(Y.shape, (20, 3))
                L = np.shape(X)[0]
                # last chunk in traj?
                last_chunk_in_traj = (
                    t + L >= reader.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= reader.number_of_trajectories() - 1)
                t += L
            # increment trajectory
            itraj += 1

    def test_stride(self):
        reader = DataInMemory(self.d)
        stride = [1, 2, 3, 4, 5, 6, 7, 10, 11, 21, 23]
        for s in stride:
            output = reader.get_output(stride=s)[0]
            expected = self.d[::s]
            np.testing.assert_allclose(output, expected,
                                       err_msg="not equal for stride=%i" % s)

    def test_chunksize(self):
        data = np.random.randn(200,2)
        cs = 100
        source = pyemma.coordinates.source(data,chunk_size=cs)
        source.chunksize = 100
        for i,ch in source.iterator():
            assert ch.shape[0] <=cs, ch.shape

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
        chunked_trajs = [[] for _ in xrange(len(data))]
        chunked_lagged_trajs = [[] for _ in xrange(len(data))]

        # iterate over data
        for itraj, X, Y in reader.iterator(lag=lag):
            chunked_trajs[itraj].append(X)
            chunked_lagged_trajs[itraj].append(Y)

        trajs = [np.vstack(ichunks) for ichunks in chunked_trajs]
        lagged_trajs = [np.vstack(ichunks) for ichunks in chunked_lagged_trajs]

        # unlagged data
        for traj, input_traj in zip(trajs, data):
            np.testing.assert_equal(traj.reshape(input_traj.shape), input_traj)

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
        chunks = [[] for _ in xrange(len(data))]
        lagged_chunks = [[] for _ in xrange(len(data))]

        # iterate over data
        for itraj, X, Y in reader.iterator(lag=lag):
            chunks[itraj].append(X)
            lagged_chunks[itraj].append(Y)

        trajs = [np.vstack(ichunks) for ichunks in chunks]

        lagged_trajs = [np.vstack(ichunks) for ichunks in lagged_chunks]

        # unlagged data
        for traj, input_traj in zip(trajs, data):
            np.testing.assert_equal(traj.reshape(input_traj.shape), input_traj)

        # lagged data
        lagged_0 = [d[lag:] for d in data]

        for traj, input_traj in zip(lagged_trajs, lagged_0):
            np.testing.assert_equal(traj.reshape(input_traj.shape), input_traj)

if __name__ == "__main__":
    unittest.main()
