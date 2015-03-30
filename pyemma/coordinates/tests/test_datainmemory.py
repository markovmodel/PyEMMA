'''
Created on 04.02.2015

@author: marscher
'''
import unittest

from pyemma.coordinates.io.data_in_memory import DataInMemory
from pyemma.util.log import getLogger
import numpy as np
from pyemma.coordinates.api import kmeans

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

        np.savetxt(f1, d)
        np.save(f2, d)

        np.savetxt(f3, d_1d)
        np.save(f4, d_1d)

        cls.files2d = [f1, f2]
        cls.files1d = [f3, f4]
        cls.d = d
        cls.d_1d = d_1d
        return cls

    @classmethod
    def tearDownClass(cls):
        for f in cls.files1d:
            os.remove(f)
        for f in cls.files2d:
            os.remove(f)

    def testListOfArrays(self):

        frames_per_traj = 100
        dim = 3
        data = [np.random.random((frames_per_traj, dim)) for i in xrange(3)]

        d = DataInMemory(data)

        self.assertEqual(d.dimension(), dim)

        self.assertEqual(
            d.trajectory_lengths(), [frames_per_traj for i in xrange(3)])

    def testDataArray(self):
        frames_per_traj = 100
        dim = 3

        data = np.random.random((frames_per_traj, dim))
        d = DataInMemory(data)

        self.assertEqual(
            d.trajectory_lengths(), [frames_per_traj for i in xrange(1)])

    def test1dData(self):
        n = 3
        data = np.arange(n)
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), [n])
        self.assertEqual(reader.dimension(), 3)
        self.assertEqual(reader.ntraj, 1)
        self.assertEqual(reader.n_frames_total(), 3)

        k = kmeans(data, k=2)
        print k.dtrajs

    def test1dDataList_diff_dim(self):
        n = 3
        data = [np.arange(n), np.arange(n + 1)]
        with self.assertRaises(ValueError):
            reader = DataInMemory(data)

    def test1dDataList(self):
        n = 10
        data = [np.arange(n), np.arange(n)]
        reader = DataInMemory(data)

        self.assertEqual(reader.trajectory_lengths(), [1, 1])
        self.assertEqual(reader.dimension(), n)
        self.assertEqual(reader.ntraj, 2)
        self.assertEqual(reader.n_frames_total(), 2)

    def test_file_1d(self):
        DataInMemory(self.files1d)

    def test_file_2d(self):
        DataInMemory(self.files2d)

    def testNotEqualDims(self):
        """ should raise, since different dims can not be processed"""
        data = [np.zeros((10, 3)), np.zeros((10, 5))]

        with self.assertRaises(ValueError):
            DataInMemory(data)

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

    def test_time_lagged_chunked_access_big_lag(self):
        n = 100
        data = [np.random.random((n, 3)), np.zeros((29, 3)),
                np.random.random((n - 50, 3))]
        reader = DataInMemory(data)
        reader.chunksize = 27

        self.assertEqual(reader.n_frames_total(), n + n - 50 + 29)

        self.assertEqual(reader.trajectory_length(0), n)
        self.assertEqual(reader.trajectory_length(1), 29)
        self.assertEqual(reader.trajectory_length(2), n-50)

        # iterate over data
        lag = 7
        t = 0
        itraj = 0
        last_chunk = False
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                X, Y = reader._next_chunk(lag=lag)
                print "X", X.shape
                print "Y", Y.shape
                print "-"*80
                #assert Y.shape[0] > 0
#                 if itraj == 0:
#                     self.assertEqual(X.shape, (100, 3))
#                     self.assertEqual(Y.shape, (70, 3))
#                 elif itraj == 1:
#                     # the time lagged chunk can not be built due to lag time
#                     self.assertEqual(X.shape, (29, 3))
#                     self.assertEqual(Y.shape, (0, 3))
#                 elif itraj == 2:
#                     self.assertEqual(X.shape, (50, 3))
#                     self.assertEqual(Y.shape, (20, 3))
                L = np.shape(X)[0]
                print "L", L
                print "itraj", itraj
                # last chunk in traj?
                last_chunk_in_traj = (
                    t + L >= reader.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= reader.number_of_trajectories() - 1)
                t += L
                print "t", t
            # increment trajectory
            itraj += 1
if __name__ == "__main__":
    unittest.main()
