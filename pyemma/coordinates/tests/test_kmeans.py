'''
Created on 28.01.2015

@author: marscher
'''
import unittest
from pyemma.coordinates.clustering.kmeans import KmeansClustering
from test_regspace import RandomDataSource


class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.k = 5
        self.kmeans = KmeansClustering(n_clusters=self.k, max_iter=100)

        self.kmeans.data_producer = RandomDataSource()

    def testName(self):
        self.kmeans.parametrize()

        assert self.kmeans.dtrajs[0].dtype == int


if __name__ == "__main__":
    unittest.main()
