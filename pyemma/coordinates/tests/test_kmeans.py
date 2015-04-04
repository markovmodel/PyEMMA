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

    def test_1d_data(self):
        # check for exception
        data = np.arange(10)

        kmeans = cluster_kmeans(data, k=7)

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
