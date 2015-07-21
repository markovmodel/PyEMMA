import unittest
from unittest import TestCase
import numpy as np
from pyemma.coordinates.api import cluster_mini_batch_kmeans
from pyemma.coordinates.api import source


class TestMiniBatchKmeans(TestCase):

    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        X = [np.random.randn(200) - 2.0,
             np.random.randn(300),
             np.random.randn(400) + 2.0]
        X = np.hstack(X)
        kmeans = cluster_mini_batch_kmeans(X, batch_size=0.5, k=10)
        cc = kmeans.clustercenters
        assert (np.any(cc < 1.0))
        assert (np.any((cc > -1.0) * (cc < 1.0)))
        assert (np.any(cc > -1.0))

    def test_3gaussian_2d_multitraj(self):
        # generate 1D data from three gaussians
        X1 = np.zeros((200, 2))
        X1[:, 0] = np.random.randn(200) - 2.0
        X2 = np.zeros((300, 2))
        X2[:, 0] = np.random.randn(300)
        X3 = np.zeros((400, 2))
        X3[:, 0] = np.random.randn(400) + 2.0
        X = [X1, X2, X3]
        kmeans = cluster_mini_batch_kmeans(X, batch_size=0.5, k=10)
        cc = kmeans.clustercenters
        assert (np.any(cc < 1.0))
        assert (np.any((cc > -1.0) * (cc < 1.0)))
        assert (np.any(cc > -1.0))

if __name__ == '__main__':
    unittest.main()
