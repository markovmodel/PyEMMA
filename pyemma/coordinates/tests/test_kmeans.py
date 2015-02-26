'''
Created on 28.01.2015

@author: marscher
'''
import unittest
from pyemma.coordinates.clustering.kmeans import KmeansClustering
from test_regspace import RandomDataSource
import tempfile
import os


class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.k = 5
        self.kmeans = KmeansClustering(n_clusters=self.k, max_iter=100)

        self.kmeans.data_producer = RandomDataSource()

    def tearDown(self):
        try:
            #os.rmdir(self.outdir)
            pass
        except:
            pass

    def testName(self):
        self.kmeans.parametrize()

        assert self.kmeans.dtrajs[0].dtype == int

    def testSaveDtrajs(self):
        self.kmeans.parametrize()
        prefix = "test"
        extension = ".dtraj"
        outdir = tempfile.mkdtemp()
        self.outdir = outdir
        try:
            self.kmeans.save_dtrajs(trajfiles=None, prefix=prefix,
                                    output_dir=outdir, extension=extension)
        except:
            os.rmdir(outdir)
            raise

        names = ["%s_%i%s" % (prefix, i, extension)
                 for i in xrange(self.kmeans.data_producer.n_samples)]
        names = [os.path.join(outdir, n) for n in names]

        # check files are there
        for f in names:
            os.stat(f)

if __name__ == "__main__":
    unittest.main()
