'''
Created on 04.02.2015

@author: marscher
'''
import unittest

from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.util.log import getLogger
import numpy as np

import tempfile
import os

logger = getLogger('TestDataInMemory')


class TestDataInMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = np.random.random((100, 3))
        d_1d = np.random.random(100)

        f1 = tempfile.mktemp()
        f2 = tempfile.mktemp(suffix='.npy')
        f3 = tempfile.mktemp()
        f4 = tempfile.mktemp(suffix='.npy')

        npz = tempfile.mktemp(suffix='.npz')

        np.savetxt(f1, d)
        np.save(f2, d)

        np.savetxt(f3, d_1d)
        np.save(f4, d_1d)

        np.savez(npz, d, d)

        cls.files2d = [f1, f2]
        cls.files1d = [f3, f4]
        cls.d = d
        cls.d_1d = d_1d

        cls.npz = npz
        return cls

    @classmethod
    def tearDownClass(cls):
        # try to clean temporary files
        try:
            for f in cls.files1d:
                os.remove(f)
            for f in cls.files2d:
                os.remove(f)

            os.remove(cls.npz)
        except:
            pass

    def testWrongArgsuments(self):
        with self.assertRaises(ValueError):
            reader = DataInMemory(self.files2d[0])

    def testListOfArrays(self):

        frames_per_traj = 100
        dim = 3
        data = [np.random.random((frames_per_traj, dim)) for _ in xrange(3)]

        d = DataInMemory(data)

        self.assertEqual(d.dimension(), dim)

        self.assertEqual(
            d.trajectory_lengths(), [frames_per_traj for _ in xrange(3)])

    def testDataArray(self):
        frames_per_traj = 100
        dim = 3

        data = np.random.random((frames_per_traj, dim))
        d = DataInMemory(data)

        self.assertEqual(
            d.trajectory_lengths(), [frames_per_traj for _ in xrange(1)])

    def test1dData(self):
        n = 3
        data = np.arange(n)
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), [n])
        self.assertEqual(reader.dimension(), 1)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.n_frames_total(), n)

    def test1dDataList(self):
        n = 10
        data = [np.arange(n), np.arange(n)]
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), [n, n])
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
        self.assertEqual(
            reader.trajectory_lengths(), [reader.n_frames_total()])

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

    def test_lagged_iterator_1d(self):
        n = 57
        chunksize = 10
        lag = 1

#         data = [np.random.random((n, 3)),
#                 np.zeros((29, 3)),
#                 np.random.random((n - 50, 3))]
#         data = [np.arange(300).reshape((n,3)),
#                 np.arange(29*3).reshape((29,3)),
#                 np.arange(150).reshape(50,3)]
        data = [np.arange(n), np.arange(50), np.arange(30)]
        input_lens = [x.shape[0] for x in data]
        # print data[0].shape
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
        n = 57
        chunksize = 10
        lag = 1

#         data = [np.random.random((n, 3)),
#                 np.zeros((29, 3)),
#                 np.random.random((n - 50, 3))]
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
