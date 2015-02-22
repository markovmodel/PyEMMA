'''
Created on 04.02.2015

@author: marscher
'''
import unittest

from pyemma.coordinates.io.data_in_memory import DataInMemory
from pyemma.util.log import getLogger
import numpy as np

logger = getLogger('TestDataInMemory')


class TestDataInMemory(unittest.TestCase):

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

    def testChunkedAccess(self):
        frames_per_traj = 100
        dim = 3
        data = [np.random.random((frames_per_traj, dim)) for i in xrange(3)]

        d = DataInMemory(data)
        d.chunksize = 6
        itraj = 0
        last_chunk = False
        while not last_chunk:
            last_chunk_in_traj = False
            t = 0
            while not last_chunk_in_traj:
                # iterate over times within trajectory
                X = d.next_chunk()
                L = np.shape(X)[0]
                # last chunk in traj?
                last_chunk_in_traj = (t + 0 + L >= d.trajectory_length(itraj))
                # last chunk?
                last_chunk = (
                    last_chunk_in_traj and itraj >= d.number_of_trajectories() - 1)
                # increment time
                t += L
            # increment trajectory
            itraj += 1

    @unittest.skip("known to be broken.")
    def testChunkedLaggedAccess(self):
        frames_per_traj = 100
        dim = 3
        lag = 3
        chunksize = 6
        data = [np.random.random((frames_per_traj, dim)) for i in xrange(3)]

        d = DataInMemory(data)
        d.chunksize = chunksize
        itraj = 0
        last_chunk = False
        ipass = 3
        while ipass > 0:
            logger.debug("ipass : %i " % ipass)
            while not last_chunk:
                last_chunk_in_traj = False
                t = 0
                while not last_chunk_in_traj:
                    # iterate over times within trajectory
                    X, Y = d.next_chunk(lag=lag)
                    assert X is not None
                    assert Y is not None
                    L = np.shape(X)[0]
                    K = np.shape(Y)[0]
                    assert L == K, "data in memory gave different chunksizes"
                    # last chunk in traj?
                    last_chunk_in_traj = (
                        t + lag + L >= d.trajectory_length(itraj))
                    # last chunk?
                    last_chunk = (
                        last_chunk_in_traj and itraj >= d.number_of_trajectories() - 1)
                    # increment time
                    t += L
                # increment trajectory
                itraj += 1
            ipass -= 1

if __name__ == "__main__":
    unittest.main()
