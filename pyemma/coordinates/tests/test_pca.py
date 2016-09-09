
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


'''
Created on 02.02.2015

@author: marscher
'''

from __future__ import absolute_import
import unittest
import os
import pkg_resources

import numpy as np

from pyemma.coordinates import pca, source
from logging import getLogger
import pyemma.util.types as types
from six.moves import range


logger = getLogger('pyemma.'+'TestPCA')


class TestPCAExtensive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import msmtools.generation as msmgen

        # set random state, remember old one and set it back in tearDownClass
        cls.old_state = np.random.get_state()
        np.random.seed(0)

        # generate HMM with two Gaussians
        cls.P = np.array([[0.99, 0.01],
                      [0.01, 0.99]])
        cls.T = 10000
        means = [np.array([-1,1]), np.array([1,-1])]
        widths = [np.array([0.3,2]),np.array([0.3,2])]
        # continuous trajectory
        cls.X = np.zeros((cls.T, 2))
        # hidden trajectory
        dtraj = msmgen.generate_traj(cls.P, cls.T)
        for t in range(cls.T):
            s = dtraj[t]
            cls.X[t,0] = widths[s][0] * np.random.randn() + means[s][0]
            cls.X[t,1] = widths[s][1] * np.random.randn() + means[s][1]
        cls.pca_obj = pca(data = cls.X, dim=1)

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls.old_state)

    def test_chunksize(self):
        assert types.is_int(self.pca_obj.chunksize)

    def test_variances(self):
        obj = pca(data = self.X)
        O = obj.get_output()[0]
        vars = np.var(O, axis=0)
        refs = obj.eigenvalues
        assert np.max(np.abs(vars - refs)) < 0.01

    def test_cumvar(self):
        assert len(self.pca_obj.cumvar) == 2
        assert np.allclose(self.pca_obj.cumvar[-1], 1.0)

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
        # Test other variants
        obj = pca(data=self.X, dim=-1, var_cutoff=1.0)
        assert obj.dimension() == 2
        obj = pca(data=self.X, dim=-1, var_cutoff=0.8)
        assert obj.dimension() == 1
        with self.assertRaises(ValueError):  # trying to set both dim and subspace_variance is forbidden
            pca(data=self.X, dim=1, var_cutoff=0.8)

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
            assert chunk.shape[1] == self.pca_obj.dimension()

    def test_map(self):
        Y = self.pca_obj.transform(self.X)
        assert Y.shape[0] == self.T
        assert Y.shape[1] == 1
        # test if consistent with get_output
        assert np.allclose(Y, self.pca_obj.get_output()[0])

    def test_mean(self):
        mean = self.pca_obj.mean
        assert len(mean) == 2
        assert np.max(mean < 0.5)

    def test_n_frames_total(self):
        # map not defined for source
        assert self.pca_obj.n_frames_total() == self.T

    def test_number_of_trajectories(self):
        # map not defined for source
        assert self.pca_obj.number_of_trajectories() == 1

    def test_output_type(self):
        assert self.pca_obj.output_type() == np.float32

    def test_trajectory_length(self):
        assert self.pca_obj.trajectory_length(0) == self.T
        with self.assertRaises(IndexError):
            self.pca_obj.trajectory_length(1)

    def test_trajectory_lengths(self):
        assert len(self.pca_obj.trajectory_lengths()) == 1
        assert self.pca_obj.trajectory_lengths()[0] == self.pca_obj.trajectory_length(0)

    def test_provided_means(self):
        data = np.random.random((300, 3))
        mean = data.mean(axis=0)
        pca_spec_mean = pca(data, mean=mean)
        pca_calc_mean = pca(data)

        np.testing.assert_allclose(mean, pca_calc_mean.mean)
        np.testing.assert_allclose(mean, pca_spec_mean.mean)

        np.testing.assert_allclose(pca_spec_mean.cov, pca_calc_mean.cov)

    def test_partial_fit(self):
        data = [np.random.random((100, 3)), np.random.random((100, 3))]
        pca_part = pca()
        pca_part.partial_fit(data[0])
        pca_part.partial_fit(data[1])

        ref = pca(data)
        np.testing.assert_allclose(pca_part.mean, ref.mean)

        np.testing.assert_allclose(pca_part.eigenvalues, ref.eigenvalues)
        np.testing.assert_allclose(pca_part.eigenvectors, ref.eigenvectors)

    def test_feature_correlation_MD(self):
        # Copying from the test_MD_data
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        self.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        self.xtc_file = os.path.join(path, 'bpti_mini.xtc')
        inp = source(self.xtc_file, top=self.pdb_file)
        pcamini = pca(inp)

        feature_traj =  pcamini.data_producer.get_output()[0]
        nfeat = feature_traj.shape[1]
        pca_traj    =  pcamini.get_output()[0]
        npcs = pca_traj.shape[1]

        test_corr = pcamini.feature_PC_correlation
        true_corr = np.corrcoef(feature_traj.T, pca_traj.T)[:nfeat,-npcs:]
        np.testing.assert_allclose(test_corr, true_corr, atol=1.E-8)

    def test_feature_correlation_data(self):
        # Create features with some correlation
        feature_traj = np.zeros((100, 3))
        feature_traj[:,0] = np.linspace(-.5,.5,len(feature_traj))
        feature_traj[:,1] = (feature_traj[:,0]+np.random.randn(len(feature_traj))*.5)**1
        feature_traj[:,2] = np.random.randn(len(feature_traj))
        nfeat = feature_traj.shape[1]

        # PCA
        pca_obj = pca(data = feature_traj, dim = 3)
        pca_traj = pca_obj.get_output()[0]
        npcs = pca_traj.shape[1]

        # Create correlations
        test_corr = pca_obj.feature_PC_correlation
        true_corr = np.corrcoef(feature_traj.T, pca_traj.T)[:nfeat,-npcs:]
        np.testing.assert_allclose(test_corr, true_corr, atol=1.E-8)

if __name__ == "__main__":
    unittest.main()
