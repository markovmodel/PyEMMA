'''
Created on 19.01.2015

@author: marscher
'''
import unittest

from pyemma.coordinates.transform.tica_amuse import Amuse

from coordinates.coordinate_transformation.discretizer import Discretizer
from coordinates.coordinate_transformation.feature_reader import FeatureReader
from coordinates.coordinate_transformation.featurizer import Featurizer
from coordinates.coordinate_transformation.tica import TICA
import numpy as np


class TestDiscretizer(unittest.TestCase):

    def setUp(self):
        """ recreate Discretizer for each test case"""
        trajfiles = ['/home/marscher/kchan/traj01_sliced.xtc']
        topfile = '/home/marscher/kchan/Traj_Structure.pdb'

        transformers = []

        # create featurizer
        featurizer = Featurizer(topfile)
        sel = np.array([(0, 20), (200, 320), (1300, 1500)])
        featurizer.distances(sel)
        # feature reader
        reader = FeatureReader(trajfiles, topfile, featurizer)

        transformers.append(reader)
        tica = TICA(reader, lag=10, output_dimension=2)
        transformers.append(tica)

        self.D = Discretizer(transformers)

    def testChunksizeResultsTica(self):
        chunk = 31
        eps = 1e-3

        self.D.run()

        # store mean and cov
        tica = self.D.transformers[-1]
        assert isinstance(tica, TICA)
        cov = tica.cov.copy()
        mean = tica.mu.copy()
        # ------- run again -------
        # reset norming factor
        tica.N = 0

        for t in self.D.transformers:
            t.parameterized = False
            t.chunksize = chunk

        self.D.run()

        np.testing.assert_allclose(tica.mu, mean, atol=eps)
        np.testing.assert_allclose(tica.cov, cov, atol=eps)

    @unittest.skip("not impled")
    def compareWithAmuse(self):
        # TODO: compare
        amuse = Amuse.compute(self.files, lag=10)
        print amuse.mean

if __name__ == "__main__":
    unittest.main()
