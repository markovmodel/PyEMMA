'''
Created on 26.01.2015

@author: marscher
'''
import itertools
import unittest
import logging

from pyemma.coordinates.clustering.regspace import RegularSpaceClustering, log
import numpy as np
import cProfile

log.setLevel(logging.ERROR)


class RandomDataSource:

    def __init__(self, chunksize=1000, a=None, b=None):
        self.n_samples = 5
        self.data = np.random.random((self.n_samples, chunksize, 3))
        if a is not None and b is not None:
            self.data *= (b - a)
            self.data += a
        self.i = -1

    def next_chunk(self, lag=0):
        self.i += 1
        return self.data[self.i]

    def reset(self):
        self.i = -1

    def trajectory_length(self, itraj):
        return self.data[itraj].shape[0]

    def number_of_trajectories(self):
        return self.data.shape[0]

    @staticmethod
    def distance(x, y):
        return np.linalg.norm(x - y, 2)

    @staticmethod
    def distances(x, Y):
        return np.linalg.norm(Y - x, 2, axis=1)


class TestRegSpaceClustering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestRegSpaceClustering, cls).setUpClass()
        np.random.seed(0)

    def setUp(self):
        self.dmin = 0.3
        self.clustering = RegularSpaceClustering(dmin=self.dmin)
        self.clustering.data_producer = RandomDataSource()
        #self.pr = cProfile.Profile()
        #self.pr.enable()
        #print "*" * 80


    def tearDown(self):
        pass
#         from pstats import Stats
#         p = Stats(self.pr)
#         p.strip_dirs()
# 
#         p.sort_stats('cumtime')
#         p.print_stats()
# 
#         print "*" * 80

    def testAlgo(self):
        self.clustering.parametrize()

        assert self.clustering.dtrajs[0].dtype == int

        # assert distance for each centroid is at least dmin
        for c in itertools.combinations(self.clustering.clustercenters, 2):
            if np.allclose(c[0], c[1]):  # skip equal pairs
                continue

            dist = np.linalg.norm(c[0] - c[1], 2)

            self.assertGreaterEqual(dist, self.dmin, "centroid pair\n%s\n%s\n has smaller"
                                    " distance than dmin(%f): %f" % (c[0], c[1], self.dmin, dist))

    def testAssignment(self):
        self.clustering.parametrize()

        assert len(self.clustering.clustercenters) > 1

        # num states == num _clustercenters?
        self.assertEqual(len(np.unique(self.clustering.dtrajs)),  len(
            self.clustering.clustercenters), "number of unique states in dtrajs"
            " should be equal.")

    def testSpreadData(self):
        self.clustering.data_producer = RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.parametrize()


if __name__ == "__main__":
    unittest.main()
