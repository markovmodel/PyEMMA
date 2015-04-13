'''
Created on 26.01.2015

@author: marscher
'''
import itertools
import unittest

from pyemma.coordinates.clustering.regspace import RegularSpaceClustering
from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates.api import cluster_regspace

import numpy as np
import pyemma.util.types as types


class RandomDataSource(DataInMemory):

    def __init__(self, a=None, b=None, chunksize=1000, n_samples=5, dim=3):
        """
        creates random values in interval [a,b]
        """
        data = np.random.random((n_samples, chunksize, dim))
        if a is not None and b is not None:
            data *= (b - a)
            data += a
        super(RandomDataSource, self).__init__(data)


class TestRegSpaceClustering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestRegSpaceClustering, cls).setUpClass()
        np.random.seed(0)

    def setUp(self):
        self.dmin = 0.3
        self.clustering = RegularSpaceClustering(dmin=self.dmin)
        self.clustering.data_producer = RandomDataSource()

    def testAlgo(self):
        self.clustering.parametrize()

        # correct type of dtrajs
        assert types.is_int_array(self.clustering.dtrajs[0])

        # assert distance for each centroid is at least dmin
        for c in itertools.combinations(self.clustering.clustercenters, 2):
            if np.allclose(c[0], c[1]):  # skip equal pairs
                continue

            dist = np.linalg.norm(c[0] - c[1], 2)

            self.assertGreaterEqual(dist, self.dmin,
                                    "centroid pair\n%s\n%s\n has smaller"
                                    " distance than dmin(%f): %f"
                                    % (c[0], c[1], self.dmin, dist))

    def testAssignment(self):
        self.clustering.parametrize()

        assert len(self.clustering.clustercenters) > 1

        # num states == num _clustercenters?
        self.assertEqual(len(np.unique(self.clustering.dtrajs)),  len(
            self.clustering.clustercenters), "number of unique states in dtrajs"
            " should be equal.")

        data_to_cluster = np.random.random((1000, 3))

        self.clustering.assign(data_to_cluster, stride=1)

    def testSpreadData(self):
        self.clustering.data_producer = RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.parametrize()

    def test1d_data(self):
        data = np.random.random(100)
        cluster_regspace(data, dmin=0.3)

if __name__ == "__main__":
    unittest.main()
