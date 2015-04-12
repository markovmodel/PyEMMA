'''
Created on 02.02.2015

@author: marscher
'''
import unittest
import os
import numpy as np

from pyemma.coordinates import source,pca
from pyemma.util.log import getLogger
import pyemma.util.types as types

logger = getLogger('TestTICA')


class TestPCA_Extensive(unittest.TestCase):

    def setUp(self):
        import pyemma.msm.generation as msmgen

        # generate HMM with two Gaussians
        self.P = np.array([[0.99, 0.01],
                      [0.01, 0.99]])
        self.T = 10000
        means = [np.array([-1,1]), np.array([1,-1])]
        widths = [np.array([0.3,2]),np.array([0.3,2])]
        # continuous trajectory
        self.X = np.zeros((self.T, 2))
        # hidden trajectory
        dtraj = msmgen.generate_traj(self.P, self.T)
        for t in range(self.T):
            s = dtraj[t]
            self.X[t,0] = widths[s][0] * np.random.randn() + means[s][0]
            self.X[t,1] = widths[s][1] * np.random.randn() + means[s][1]
        self.lag = 10
        self.pca_obj = pca(data = self.X, dim=1)

    def test_chunksize(self):
        assert types.is_int(self.pca_obj.chunksize)

    def test_cov(self):
        cov_ref = np.dot(self.X.T, self.X) / float(self.T)
        assert(np.all(self.pca_obj.cov.shape == cov_ref.shape))
        assert(np.max(self.pca_obj.cov - cov_ref) < 3e-2)

    def test_data_producer(self):
        assert self.pca_obj.data_producer is not None

    def test_describe(self):
        desc = self.pca_obj.describe()
        assert types.is_string(desc) or types.is_list_of_string(desc)

    def test_dimension(self):
        assert types.is_int(self.pca_obj.dimension())
        # Here:
        assert self.pca_obj.dimension() == 1

    def test_eigenvalues(self):
        eval = self.pca_obj.eigenvalues
        assert len(eval) == 2

    def test_eigenvectors(self):
        evec = self.pca_obj.eigenvectors
        assert(np.all(evec.shape == (2,2)))

    def test_get_output(self):
        O = self.pca_obj.get_output()
        assert types.is_list(O)
        assert len(O) == 1
        assert types.is_float_matrix(O[0])
        assert O[0].shape[0] == self.T
        assert O[0].shape[1] == self.pca_obj.dimension()

    def test_in_memory(self):
        assert isinstance(self.pca_obj.in_memory, bool)

    def test_iterator(self):
        for itraj, chunk in self.pca_obj:
            assert types.is_int(itraj)
            assert types.is_float_matrix(chunk)
            assert chunk.shape[0] <= self.pca_obj.chunksize + self.lag
            assert chunk.shape[1] == self.pca_obj.dimension()

    def test_map(self):
        Y = self.pca_obj.map(self.X)
        assert Y.shape[0] == self.T
        assert Y.shape[1] == 1
        # test if consistent with get_output
        assert np.allclose(Y, self.pca_obj.get_output()[0])

    def test_mean(self):
        mean = self.pca_obj.mean
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_mu(self):
        mean = self.pca_obj.mu
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_n_frames_total(self):
        # map not defined for source
        self.pca_obj.n_frames_total() == self.T

    def test_number_of_trajectories(self):
        # map not defined for source
        self.pca_obj.number_of_trajectories() == 1

    def test_output_type(self):
        assert self.pca_obj.output_type() == np.float32

    def test_parametrize(self):
        # nothing should happen
        self.pca_obj.parametrize()

    def test_trajectory_length(self):
        assert self.pca_obj.trajectory_length(0) == self.T
        with self.assertRaises(IndexError):
            self.pca_obj.trajectory_length(1)

    def test_trajectory_lengths(self):
        assert len(self.pca_obj.trajectory_lengths()) == 1
        assert self.pca_obj.trajectory_lengths()[0] == self.pca_obj.trajectory_length(0)

if __name__ == "__main__":
    unittest.main()
