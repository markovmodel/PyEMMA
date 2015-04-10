'''
Created on 28.01.2015

@author: marscher
'''
import unittest
import tempfile
import os
import numpy as np
from pyemma.coordinates.api import cluster_kmeans
import shutil


class TestKmeans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestKmeans, cls).setUpClass()
        cls.dtraj_dir = tempfile.mkdtemp()

    def setUp(self):
        self.k = 5
        self.dim = 100
        self.data = [np.random.random((30, self.dim)),
                     np.random.random((37, self.dim))]
        self.kmeans = cluster_kmeans(data=self.data, k=self.k, max_iter=100)

    def tearDown(self):
        shutil.rmtree(self.dtraj_dir, ignore_errors=True)

    def testDtraj(self):
        assert self.kmeans.dtrajs[0].dtype == int

    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        X = [np.random.randn(100)-2.0,
             np.random.randn(100),
             np.random.randn(100)+2.0]
        X = np.hstack(X)
        kmeans = cluster_kmeans(X, k=10)
        cc = kmeans.clustercenters
        assert(np.any(cc < 1.0))
        assert(np.any((cc > -1.0) * (cc < 1.0)))
        assert(np.any(cc > -1.0))

    def test_3gaussian_2d_multitraj(self):
        # generate 1D data from three gaussians
        X1 = np.zeros((100,2))
        X1[:,0] = np.random.randn(100)-2.0
        X2 = np.zeros((100,2))
        X2[:,0] = np.random.randn(100)
        X3 = np.zeros((100,2))
        X3[:,0] = np.random.randn(100)+2.0
        X = [X1, X2, X3]
        kmeans = cluster_kmeans(X, k=10)
        cc = kmeans.clustercenters
        assert(np.any(cc < 1.0))
        assert(np.any((cc > -1.0) * (cc < 1.0)))
        assert(np.any(cc > -1.0))

    def testSaveDtrajs(self):
        prefix = "test"
        extension = ".dtraj"
        outdir = self.dtraj_dir
        self.kmeans.save_dtrajs(trajfiles=None, prefix=prefix,
                                    output_dir=outdir, extension=extension)

        names = ["%s_%i%s" % (prefix, i, extension)
                 for i in xrange(self.kmeans.data_producer.number_of_trajectories())]
        names = [os.path.join(outdir, n) for n in names]

        # check files with given patterns are there
        for f in names:
            os.stat(f)

if __name__ == "__main__":
    unittest.main()
