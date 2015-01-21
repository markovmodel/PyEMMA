'''
Created on 19.01.2015

@author: marscher
'''
import unittest
from coordinates.coordinate_transformation.discretizer import Discretizer
from coordinates.coordinate_transformation.tica import TICA

from pyemma.coordinates.transform.tica_amuse import Amuse

import numpy as np

# TODO: clean up
D = None
runned_discretizer = False
files = ['/home/marscher/kchan/traj01_sliced.xtc']
topo = '/home/marscher/kchan/Traj_Structure.pdb'


def run_discretizer():
    global D
    if not runned_discretizer:
        D = Discretizer(files, topo)
        D.run()


class TestDiscretizer(unittest.TestCase):

    def testChunksizeResultsTica(self):
        chunk = 3
        eps = 1e-3
        run_discretizer()

        # store mean and cov
        tica = D.transformers[-1]
        assert isinstance(tica, TICA)
        cov = tica.cov.copy()
        mean = tica.mu.copy()
        # ------- run again -------
        # reset norming factor
        tica.N = 0

        for t in D.transformers:
            t.parameterized = False
            t.set_chunksize(chunk)

        D.run()

        np.testing.assert_allclose(tica.mu, mean, atol=eps)
        np.testing.assert_allclose(tica.cov, cov, atol=eps)

    @unittest.skip("not impled")
    def compareWithAmuse(self):
        # TODO: compare
        amuse = Amuse.compute(self.files, lag=10)
        print amuse.mean

if __name__ == "__main__":
    unittest.main()
