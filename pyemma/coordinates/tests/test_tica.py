"""
Created on 02.02.2015

@author: marscher
"""
import unittest
import os
import numpy as np

from pyemma.coordinates import api

from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates import source, tica
from pyemma.util.log import getLogger
import pyemma.util.types as types

logger = getLogger('TestTICA')


class TestTICA_Basic(unittest.TestCase):
    def test(self):
        np.random.seed(0)

        data = np.random.randn(100, 10)
        tica_obj = api.tica(data=data, lag=10, dim=1)
        tica_obj.parametrize()
        Y = tica_obj._map_array(data)
        # right shape
        assert types.is_float_matrix(Y)
        assert Y.shape[0] == 100
        assert Y.shape[1] == 1

    def test_MD_data(self):
        # this is too little data to get reasonable results. We just test to avoid exceptions
        path = os.path.join(os.path.split(__file__)[0], 'data')
        self.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        self.xtc_file = os.path.join(path, 'bpti_mini.xtc')
        inp = source(self.xtc_file, top=self.pdb_file)
        # see if this doesn't raise
        ticamini = tica(inp, lag=1)

    def test_duplicated_data(self):
        # make some data that has one column repeated twice
        X = np.random.randn(100, 2)
        X = np.hstack((X, X[:, 0, np.newaxis]))

        d = DataInMemory(X)

        tica_obj = api.tica(data=d, lag=1, dim=1)

        assert tica_obj.eigenvectors.dtype == np.float64
        assert tica_obj.eigenvalues.dtype == np.float64

    def test_singular_zeros(self):
        # make some data that has one column of all zeros
        X = np.random.randn(100, 2)
        X = np.hstack((X, np.zeros((100, 1))))

        tica_obj = api.tica(data=X, lag=1, dim=1)

        assert tica_obj.eigenvectors.dtype == np.float64
        assert tica_obj.eigenvalues.dtype == np.float64

    def testChunksizeResultsTica(self):
        chunk = 40
        lag = 100
        np.random.seed(0)
        X = np.random.randn(23000, 3)

        # un-chunked
        d = DataInMemory(X)

        tica_obj = api.tica(data=d, lag=lag, dim=1)

        cov = tica_obj.cov.copy()
        mean = tica_obj.mu.copy()

        # ------- run again with new chunksize -------
        d = DataInMemory(X)
        d.chunksize = chunk
        tica_obj = tica(data=d, lag=lag, dim=1)

        np.testing.assert_allclose(tica_obj.mu, mean)
        np.testing.assert_allclose(tica_obj.cov, cov)


class TestTICAExtensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyemma.msm.generation as msmgen

        # generate HMM with two Gaussians
        cls.P = np.array([[0.99, 0.01],
                          [0.01, 0.99]])
        cls.T = 10000
        means = [np.array([-1, 1]), np.array([1, -1])]
        widths = [np.array([0.3, 2]), np.array([0.3, 2])]
        # continuous trajectory
        cls.X = np.zeros((cls.T, 2))
        # hidden trajectory
        dtraj = msmgen.generate_traj(cls.P, cls.T)
        for t in range(cls.T):
            s = dtraj[t]
            cls.X[t, 0] = widths[s][0] * np.random.randn() + means[s][0]
            cls.X[t, 1] = widths[s][1] * np.random.randn() + means[s][1]
        cls.lag = 10
        cls.tica_obj = api.tica(data=cls.X, lag=cls.lag, dim=1)

    def setUp(self):
        pass

    def test_chunksize(self):
        assert types.is_int(self.tica_obj.chunksize)

    def test_cov(self):
        cov_ref = np.dot(self.X.T, self.X) / float(self.T)
        assert (np.all(self.tica_obj.cov.shape == cov_ref.shape))
        assert (np.max(self.tica_obj.cov - cov_ref) < 3e-2)

    def test_cov_tau(self):
        cov_tau_ref = np.dot(self.X[self.lag:].T, self.X[:self.T - self.lag]) / float(self.T - self.lag)
        assert (np.all(self.tica_obj.cov_tau.shape == cov_tau_ref.shape))
        assert (np.max(self.tica_obj.cov_tau - cov_tau_ref) < 3e-2)

    def test_data_producer(self):
        assert self.tica_obj.data_producer is not None

    def test_describe(self):
        desc = self.tica_obj.describe()
        assert types.is_string(desc) or types.is_list_of_string(desc)

    def test_dimension(self):
        assert types.is_int(self.tica_obj.dimension())
        # Here:
        assert self.tica_obj.dimension() == 1

    def test_eigenvalues(self):
        eval = self.tica_obj.eigenvalues
        assert len(eval) == 2
        assert np.max(np.abs(eval) < 1)

    def test_eigenvectors(self):
        evec = self.tica_obj.eigenvectors
        assert (np.all(evec.shape == (2, 2)))
        assert np.max(np.abs(evec[:, 0]) - np.array([1, 0]) < 0.05)

    def test_get_output(self):
        O = self.tica_obj.get_output()
        assert types.is_list(O)
        assert len(O) == 1
        assert types.is_float_matrix(O[0])
        assert O[0].shape[0] == self.T
        assert O[0].shape[1] == self.tica_obj.dimension()

    def test_in_memory(self):
        assert isinstance(self.tica_obj.in_memory, bool)

    def test_iterator(self):
        for itraj, chunk in self.tica_obj:
            assert types.is_int(itraj)
            assert types.is_float_matrix(chunk)
            assert chunk.shape[0] <= self.tica_obj.chunksize + self.lag
            assert chunk.shape[1] == self.tica_obj.dimension()

    def test_lag(self):
        assert types.is_int(self.tica_obj.lag)
        # Here:
        assert self.tica_obj.lag == self.lag

    def test_map(self):
        Y = self.tica_obj.map(self.X)
        assert Y.shape[0] == self.T
        assert Y.shape[1] == 1
        # test if consistent with get_output
        assert np.allclose(Y, self.tica_obj.get_output()[0])

    def test_mean(self):
        mean = self.tica_obj.mean
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_mu(self):
        mean = self.tica_obj.mu
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_n_frames_total(self):
        # map not defined for source
        self.tica_obj.n_frames_total() == self.T

    def test_number_of_trajectories(self):
        # map not defined for source
        self.tica_obj.number_of_trajectories() == 1

    def test_output_type(self):
        assert self.tica_obj.output_type() == np.float32

    def test_parametrize(self):
        # nothing should happen
        self.tica_obj.parametrize()

    def test_trajectory_length(self):
        assert self.tica_obj.trajectory_length(0) == self.T
        with self.assertRaises(IndexError):
            self.tica_obj.trajectory_length(1)

    def test_trajectory_lengths(self):
        assert len(self.tica_obj.trajectory_lengths()) == 1
        assert self.tica_obj.trajectory_lengths()[0] == self.tica_obj.trajectory_length(0)


if __name__ == "__main__":
    unittest.main()
