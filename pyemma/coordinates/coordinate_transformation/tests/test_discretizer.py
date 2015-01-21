'''
Created on 19.01.2015

@author: marscher
'''
import unittest
from coordinates.coordinate_transformation.discretizer import Discretizer
from coordinates.coordinate_transformation.tica import TICA

import numpy as np


class TestDiscretizer(unittest.TestCase):

    def testChunksizeResultsTica(self):
        chunk = 50
        eps = 1e-4

        d = Discretizer(['/home/marscher/kchan/traj01_sliced.xtc'],
                        # '/home/marscher/kchan/Traj01.xtc'],
                        #  '/home/marscher/kchan/Traj_link.xtc'],
                        '/home/marscher/kchan/Traj_Structure.pdb')
        # run once with calculated chunk size
        d.run()

        # store mean and cov
        tica = d.transformers[-1]
        assert isinstance(tica, TICA)
        cov = tica.cov.copy()
        mean = tica.mu.copy()
        # reset norming factor
        tica.N = 0

        for t in d.transformers:
            t.parameterized = False
            t.set_chunksize(chunk)

        d.run()

        np.testing.assert_allclose(tica.mu, mean, atol=eps)
        np.testing.assert_allclose(tica.cov, cov, atol=eps)


if __name__ == "__main__":
    unittest.main()
