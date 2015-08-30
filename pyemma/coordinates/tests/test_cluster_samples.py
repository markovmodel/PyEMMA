
"""
Test the save_trajs function of the coordinates API by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
"""

from __future__ import absolute_import

import unittest

import numpy as np
import pyemma.coordinates as coor

class TestClusterSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestClusterSamples, cls).setUpClass()

    def setUp(self):
        self.input_trajs = [[0,1,2],
                            [3,4,5],
                            [6,7,8],
                            [0,1,2],
                            [3,4,5],
                            [6,7,8]]
        self.cluster_obj = coor.cluster_regspace(data=self.input_trajs, dmin=.5)

    def test_index_states(self):
        # Test that the catalogue is being set up properly

        # The assingment-catalogue is easy to see from the above dtrajs
        ref = [[[0,0],[3,0]], # appearances of the 1st cluster
               [[0,1],[3,1]], # appearances of the 2nd cluster
               [[0,2],[3,2]], # appearances of the 3rd cluster
               [[1,0],[4,0]], #    .....
               [[1,1],[4,1]],
               [[1,2],[4,2]],
               [[2,0],[5,0]],
               [[2,1],[5,1]],
               [[2,2],[5,2]],
               ]

        for cc in np.arange(self.cluster_obj.n_clusters):
            assert np.allclose(self.cluster_obj.index_clusters[cc], ref[cc])

    def test_sample_indexes_by_state(self):
        samples = self.cluster_obj.sample_indexes_by_cluster(np.arange(self.cluster_obj.n_clusters), 10)

        # For each sample, check that you're actually retrieving the i-th center
        for ii, isample in enumerate(samples):
            assert np.in1d([self.cluster_obj.dtrajs[pair[0]][pair[1]] for pair in isample],ii).all()


if __name__ == "__main__":
    unittest.main()
