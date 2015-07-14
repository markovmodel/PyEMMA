__author__ = 'mho'

from unittest import TestCase
import numpy as np
from pyemma.coordinates.api import cluster_mini_batch_kmeans

class TestMiniBatchKmeans(TestCase):

    def setUp(self):
        self.k = 2
        self.dim=100
        self.data = [np.random.random((30, self.dim)),
                     np.random.random((37, self.dim))]
        self.kmeans = cluster_mini_batch_kmeans(self.data, k=self.k)

    def test_cluster_centers(self):
        cc = self.kmeans.clustercenters
        print cc