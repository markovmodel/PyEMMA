
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 02.02.2015

@author: marscher
'''
import unittest

import numpy as np

from pyemma.coordinates import pca
from pyemma.util.log import getLogger
import pyemma.util.types as types


logger = getLogger('TestTICA')


class TestPCAExtensive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.msm.generation as msmgen

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
        cls.lag = 10
        cls.pca_obj = pca(data = cls.X, dim=1)

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls.old_state)

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
        assert self.pca_obj.n_frames_total() == self.T

    def test_number_of_trajectories(self):
        # map not defined for source
        assert self.pca_obj.number_of_trajectories() == 1

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