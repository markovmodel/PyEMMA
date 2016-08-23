
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Created on 02.02.2015

@author: marscher
"""

from __future__ import absolute_import
import unittest
import os
import pkg_resources
import numpy as np

from pyemma.coordinates import api

from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates import source, tica
from pyemma.coordinates.transform import TICA as _internal_tica
from pyemma.util.contexts import numpy_random_seed
from logging import getLogger
import pyemma.util.types as types
from six.moves import range

logger = getLogger('pyemma.'+'TestTICA')


def mycorrcoef(X, Y, lag):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    mean_X = 0.5*(np.mean(X[lag:], axis=0)+np.mean(X[0:-lag], axis=0))
    mean_Y = 0.5*(np.mean(Y[lag:], axis=0)+np.mean(Y[0:-lag], axis=0))
    cov = ((X[0:-lag]-mean_X).T.dot(Y[0:-lag]-mean_Y) +
           (X[lag:]-mean_X).T.dot(Y[lag:]-mean_Y)) / (2*(X.shape[0]-lag)-1)

    autocov_X = ((X[0:-lag]-mean_X).T.dot(X[0:-lag]-mean_X) +
                 (X[lag:]-mean_X).T.dot(X[lag:]-mean_X)) / (2*(X.shape[0]-lag)-1)
    var_X = np.diag(autocov_X)
    autocov_Y = ((Y[0:-lag]-mean_Y).T.dot(Y[0:-lag]-mean_Y) +
                 (Y[lag:]-mean_Y).T.dot(Y[lag:]-mean_Y)) / (2*(Y.shape[0]-lag)-1)
    var_Y = np.diag(autocov_Y)
    return cov / np.sqrt(var_X[:,np.newaxis]) / np.sqrt(var_Y)


class TestTICA_Basic(unittest.TestCase):
    def test(self):
        # make it deterministic
        with numpy_random_seed(0):
            data = np.random.randn(100, 10)
        tica_obj = api.tica(data=data, lag=10, dim=1)
        tica_obj.parametrize()
        Y = tica_obj._transform_array(data)
        # right shape
        assert types.is_float_matrix(Y)
        assert Y.shape[0] == 100
        assert Y.shape[1] == 1, Y.shape[1]

    def test_MD_data(self):
        # this is too little data to get reasonable results. We just test to avoid exceptions
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
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

    def test_fit_transform(self):
        X = np.random.randn(100, 2)
        tica = _internal_tica(1, 1)
        out = tica.fit_transform(X)
        np.testing.assert_array_almost_equal(out, api.tica(data=X, lag=1, dim=1).get_output()[0])

    def test_duplicated_data_in_fit_transform(self):
        X = np.random.randn(100, 2)
        d = DataInMemory([X, X])
        tica = api.tica(data=d, lag=1, dim=1)
        out1 = tica.get_output()
        out2 = tica.fit_transform([X, X])
        np.testing.assert_array_almost_equal(out1, out2)


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
        mean = tica_obj.mean.copy()

        # ------- run again with new chunksize -------
        d = DataInMemory(X)
        d.chunksize = chunk
        tica_obj = tica(data=d, lag=lag, dim=1)

        np.testing.assert_allclose(tica_obj.mean, mean)
        np.testing.assert_allclose(tica_obj.cov, cov)

    def test_in_memory(self):
        data = np.random.random((100, 10))
        tica_obj = api.tica(lag=10, dim=1)
        reader = api.source(data)
        tica_obj.data_producer = reader

        tica_obj.in_memory = True
        tica_obj.parametrize()
        tica_obj.get_output()

    def test_too_short_trajs(self):
        trajs = [np.empty((100, 1))]
        with self.assertRaises(ValueError):
            tica(trajs, lag=100)

    def test_with_skip(self):
        data = np.random.random((100, 10))
        tica_obj = api.tica(lag=10, dim=1, skip=1)


class TestTICAExtensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with numpy_random_seed(123):
            import msmtools.generation as msmgen

            # generate HMM with two Gaussians
            cls.P = np.array([[0.99, 0.01],
                              [0.01, 0.99]])
            cls.T = 40000
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
            # do unscaled TICA
            reader=api.source(cls.X, chunk_size=0)
            cls.tica_obj = api.tica(data=reader, lag=cls.lag, dim=1, kinetic_map=False)

    def setUp(self):
        pass

    def test_variances(self):
        # test unscaled TICA:
        O = self.tica_obj.get_output()[0]
        vars = np.var(O, axis=0)
        assert np.max(np.abs(vars - 1.0)) < 0.01

    def test_kinetic_map(self):
        # test kinetic map variances:
        tica_kinmap = api.tica(data=self.X, lag=self.lag, dim=-1,var_cutoff=1, kinetic_map=True)
        O = tica_kinmap.get_output()[0]
        vars = np.var(O, axis=0)
        refs = tica_kinmap.eigenvalues ** 2
        assert np.max(np.abs(vars - refs)) < 0.01

    def test_cumvar(self):
        assert len(self.tica_obj.cumvar) == 2
        assert np.allclose(self.tica_obj.cumvar[-1], 1.0)

    def test_chunksize(self):
        assert types.is_int(self.tica_obj.chunksize)

    def test_cov(self):
        cov_ref = np.dot(self.X.T, self.X) / float(self.T)
        assert (np.all(self.tica_obj.cov.shape == cov_ref.shape))
        np.testing.assert_allclose(self.tica_obj.cov, cov_ref, atol=5e-2)

    def test_cov_tau(self):
        cov_tau_ref = np.dot(self.X[self.lag:].T, self.X[:self.T - self.lag]) / float(self.T - self.lag)
        cov_tau_ref = 0.5 * (cov_tau_ref + cov_tau_ref.T)
        assert (np.all(self.tica_obj.cov_tau.shape == cov_tau_ref.shape))
        assert (np.max(self.tica_obj.cov_tau - cov_tau_ref) < 5e-2)

    def test_data_producer(self):
        assert self.tica_obj.data_producer is not None

    def test_describe(self):
        desc = self.tica_obj.describe()
        assert types.is_string(desc) or types.is_list_of_string(desc)

    def test_dimension(self):
        assert types.is_int(self.tica_obj.dimension())
        # Here:
        assert self.tica_obj.dimension() == 1
        # Test other variants
        tica = api.tica(data=self.X, lag=self.lag, dim=-1, var_cutoff=1.0)
        assert tica.dimension() == 2
        tica = api.tica(data=self.X, lag=self.lag, dim=-1, var_cutoff=0.9)
        assert tica.dimension() == 1
        with self.assertRaises(ValueError):  # trying to set both dim and subspace_variance is forbidden
            api.tica(data=self.X, lag=self.lag, dim=1, var_cutoff=0.9)

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
        chunksize=1000
        for itraj, chunk in self.tica_obj.iterator(chunk=chunksize):
            assert types.is_int(itraj)
            assert types.is_float_matrix(chunk)
            self.assertLessEqual(chunk.shape[0], (chunksize + self.lag))
            assert chunk.shape[1] == self.tica_obj.dimension()

    def test_lag(self):
        assert types.is_int(self.tica_obj.lag)
        # Here:
        assert self.tica_obj.lag == self.lag

    def test_map(self):
        Y = self.tica_obj.transform(self.X)
        assert Y.shape[0] == self.T
        assert Y.shape[1] == 1
        # test if consistent with get_output
        assert np.allclose(Y, self.tica_obj.get_output()[0])

    def test_mean(self):
        mean = self.tica_obj.mean
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_mu(self):
        mean = self.tica_obj.mean
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

    def test_feature_correlation_MD(self):
        # Copying from the test_MD_data
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        self.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        self.xtc_file = os.path.join(path, 'bpti_mini.xtc')
        inp = source(self.xtc_file, top=self.pdb_file)
        ticamini = tica(inp, lag=1, kinetic_map=False)

        feature_traj =  ticamini.data_producer.get_output()[0]
        tica_traj    =  ticamini.get_output()[0]
        test_corr = ticamini.feature_TIC_correlation
        true_corr = mycorrcoef(feature_traj, tica_traj, ticamini.lag)
        #assert np.isclose(test_corr, true_corr).all()
        np.testing.assert_allclose(test_corr, true_corr, atol=1.E-8)

    def test_feature_correlation_data(self):
        # Create features with some correlation
        feature_traj = np.zeros((100, 3))
        feature_traj[:,0] = np.linspace(-.5,.5,len(feature_traj))
        feature_traj[:,1] = (feature_traj[:,0]+np.random.randn(len(feature_traj))*.5)**1
        feature_traj[:,2] = np.random.randn(len(feature_traj))

        # Tica
        tica_obj = tica(data = feature_traj, dim = 3, kinetic_map=False)
        tica_traj = tica_obj.get_output()[0]

        # Create correlations
        test_corr = tica_obj.feature_TIC_correlation
        true_corr = mycorrcoef(feature_traj, tica_traj, tica_obj.lag)
        np.testing.assert_allclose(test_corr, true_corr, atol=1.E-8)
        #assert np.isclose(test_corr, true_corr).all()

    def test_provided_means(self):
        data = np.random.random((300, 3))
        mean = data.mean(axis=0)
        tica_obj = tica(data, mean=mean)
        tica_calc_mean = tica(data)

        np.testing.assert_allclose(tica_obj.mean, tica_calc_mean.mean)
        np.testing.assert_allclose(tica_obj.cov, tica_calc_mean.cov)
        np.testing.assert_allclose(tica_obj.cov_tau, tica_calc_mean.cov_tau)

    def test_timescales(self):
        its = -self.tica_obj.lag/np.log(np.abs(self.tica_obj.eigenvalues))
        assert np.allclose(self.tica_obj.timescales, its)

    def test_too_short_traj_partial_fit(self):
        data = [np.empty((20, 3)), np.empty((10, 3))]
        lag = 11
        tica_obj = tica(lag=lag)
        from pyemma.util.testing_tools import MockLoggingHandler
        log_handler = MockLoggingHandler()
        tica_obj.logger.addHandler(log_handler)
        for x in data:
            tica_obj.partial_fit(x)

        self.assertEqual(tica_obj._used_data, 20 - lag)
        self.assertEqual(len(log_handler.messages['warning']), 1)
        self.assertIn("longer than lag time", log_handler.messages['warning'][0])


if __name__ == "__main__":
    unittest.main()
