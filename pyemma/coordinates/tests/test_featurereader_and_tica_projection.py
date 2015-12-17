
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
Test feature reader and Tica by checking the properties of the ICs.
cov(ic_i,ic_j) = delta_ij and cov(ic_i,ic_j,tau) = lambda_i delta_ij
@author: Fabian Paul
'''

from __future__ import print_function

from __future__ import absolute_import
import unittest
import os
import tempfile
import numpy as np
import mdtraj
from pyemma.coordinates.api import tica, _TICA as TICA
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.util.log import getLogger
from six.moves import range

log = getLogger('pyemma.'+'TestFeatureReaderAndTICAProjection')

def random_invertible(n, eps=0.01):
    'generate real random invertible matrix'
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    s = np.maximum(s, eps)
    return u.dot(np.diag(s)).dot(v)

from nose.plugins.attrib import attr


@attr(slow=True)
class TestFeatureReaderAndTICAProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReaderAndTICAProjection, cls).setUpClass()

        cls.dim = 99  # dimension (must be divisible by 3)
        N = 5000  # length of single trajectory # 500000
        N_trajs = 10  # number of trajectories

        A = random_invertible(cls.dim)  # mixing matrix
        # tica will approximate its inverse with the projection matrix
        mean = np.random.randn(cls.dim)

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb, 'w') as f:
            for i in range(cls.dim // 3):
                print(('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i), file=f)

        t = np.arange(0, N)
        cls.trajnames = []  # list of xtc file names
        for i in range(N_trajs):
            # set up data
            white = np.random.randn(N, cls.dim)
            brown = np.cumsum(white, axis=0)
            correlated = np.dot(brown, A)
            data = correlated + mean
            xyz = data.reshape((N, cls.dim // 3, 3))
            # create trajectory file
            traj = mdtraj.load(cls.temppdb)
            traj.xyz = xyz
            traj.time = t
            tempfname = tempfile.mktemp('.xtc')
            traj.save(tempfname)
            cls.trajnames.append(tempfname)

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestFeatureReaderAndTICAProjection, cls).tearDownClass()
        
    def test_covariances_and_eigenvalues(self):
        reader = FeatureReader(self.trajnames, self.temppdb)
        for tau in [1, 10, 100, 1000, 2000]:
            trans = TICA(lag=tau, kinetic_map=False, force_eigenvalues_le_one=True, var_cutoff=1.0)
            trans.data_producer = reader

            log.info('number of trajectories reported by tica %d' % trans.number_of_trajectories())
            trans.parametrize()
            data = trans.get_output()
            # print '@@cov', trans.cov
            # print '@@cov_tau', trans.cov_tau

            log.info('max. eigenvalue: %f' % np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))
            # check ICs
            check = tica(data=data, lag=tau, dim=self.dim, force_eigenvalues_le_one=True)
            check.parametrize()

            np.testing.assert_allclose(np.eye(self.dim), check.cov, atol=1e-8)
            np.testing.assert_allclose(check.mu, 0.0, atol=1e-8)
            ic_cov_tau = np.zeros((self.dim, self.dim))
            ic_cov_tau[np.diag_indices(self.dim)] = trans.eigenvalues
            np.testing.assert_allclose(ic_cov_tau, check.cov_tau, atol=1e-8)
            # print '@@cov_tau', check.cov_tau

if __name__ == "__main__":
    unittest.main()
