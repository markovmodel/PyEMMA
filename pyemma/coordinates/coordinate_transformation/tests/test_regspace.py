'''
Created on 26.01.2015

@author: marscher
'''
import itertools
import unittest

from coordinates.coordinate_transformation.regspace_clustering import RegularSpaceClustering
import numpy as np


#import matplotlib.pyplot as plt


class RandomDataSource:

    def __init__(self, chunksize=100):
        self.n_samples = 100
        self.data = np.random.random((self.n_samples, chunksize, 2))
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

    def setUp(self):
        source = RandomDataSource()
        print "n  trajs: ", source.number_of_trajectories()
        self.dmin = 0.7
        self.clustering = RegularSpaceClustering(source, dmin=self.dmin)

    def testAlgo(self):
        self.clustering.parametrize()

        # plt.plot(self.clustering._centroids)

        # assert distance for each centroid is at least dmin
        for c in itertools.combinations(self.clustering._centroids, 2):
            if np.allclose(c[0], c[1]):  # skip equal pairs
                continue

            dist = np.linalg.norm(c[0] - c[1], 2)

            self.assertGreaterEqual(dist, self.dmin, "centroid pair\n%s\n%s\n has smaller"
                                    " distance than dmin(%f): %f" % (c[0], c[1], self.dmin, dist))

    def testAssignment(self):
        self.clustering.parametrize()

        # num states == num centroids?
        assert len(np.unique(self.clustering.dtrajs)) == len(
            self.clustering._centroids)
        
        #for each
        

if __name__ == "__main__":
    unittest.main()
