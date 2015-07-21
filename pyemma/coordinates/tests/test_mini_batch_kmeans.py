import unittest
from unittest import TestCase
import numpy as np
from pyemma.coordinates.api import cluster_mini_batch_kmeans
from pyemma.coordinates.api import source


class TestMiniBatchKmeans(TestCase):
    def setUp(self):
        self.k = 20
        self.dim = 5
        self.data = [np.random.random((600, self.dim)),
                     np.random.random((100, self.dim)),
                     np.random.random((2, self.dim))]

    def test_cluster_centers(self):
        # data in memory: complete trajectory mode
        kmeans = cluster_mini_batch_kmeans(source(self.data, chunk_size=0), k=self.k, batch_size=0.5)
        # data in memory: chunked access
        #kmeans = cluster_mini_batch_kmeans(source(self.data, chunk_size=3000), k=self.k, batch_size=0.5)

        cc = kmeans.clustercenters
        print cc

    def test_cluster_centers_in_memory(self):
        pass
        # kmeans = cluster_mini_batch_kmeans(self.data, k=self.k, batch_size=0.5)
        # kmeans.in_memory = True
        # cc = kmeans.clustercenters
        # print cc
        # for c in range(1, 8):
        #     kmeans.chunksize=c
        #     for x in kmeans.iterator(stride={1: [1,2,3,4,5,6], 2: [1]}):
        #         print x
        #     print "***"*15


if __name__ == '__main__':
    unittest.main()
