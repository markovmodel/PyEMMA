
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

from __future__ import absolute_import
from __future__ import print_function

import os
import tempfile
import unittest

from nose.plugins.attrib import attr
import mdtraj

from pyemma.coordinates.api import tica
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.util.contexts import numpy_random_seed
from logging import getLogger
from six.moves import range
import numpy as np


log = getLogger('pyemma.'+'TestFeatureReaderAndTICAProjection')


def random_invertible(n, eps=0.01):
    'generate real random invertible matrix'
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    s = np.maximum(s, eps)
    return u.dot(np.diag(s)).dot(v)


@attr(slow=True)
class TestFeatureReaderAndTICAProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with numpy_random_seed(52):
            c = super(TestFeatureReaderAndTICAProjection, cls).setUpClass()

            cls.dim = 99  # dimension (must be divisible by 3)
            N = 5000  # length of single trajectory # 500000 # 50000
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
            trans = tica(lag=tau, dim=self.dim, kinetic_map=False)
            trans.data_producer = reader

            log.info('number of trajectories reported by tica %d' % trans.number_of_trajectories())
            trans.parametrize()
            data = trans.get_output()

            log.info('max. eigenvalue: %f' % np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))
            # check ICs
            check = tica(data=data, lag=tau, dim=self.dim)

            np.testing.assert_allclose(np.eye(self.dim), check.cov, atol=1e-8)
            np.testing.assert_allclose(check.mean, 0.0, atol=1e-8)
            ic_cov_tau = np.zeros((self.dim, self.dim))
            ic_cov_tau[np.diag_indices(self.dim)] = trans.eigenvalues
            np.testing.assert_allclose(ic_cov_tau, check.cov_tau, atol=1e-8)

    def test_partial_fit(self):
        from pyemma.coordinates import source
        reader = source(self.trajnames, top=self.temppdb)
        reader_output = reader.get_output()

        params = {'lag': 10, 'kinetic_map': False, 'dim': self.dim}

        tica_obj = tica(**params)
        tica_obj.partial_fit(reader_output[0])
        assert not tica_obj._estimated
        # acccess eigenvectors to force diagonalization
        tica_obj.eigenvectors
        assert tica_obj._estimated

        tica_obj.partial_fit(reader_output[1])
        assert not tica_obj._estimated

        tica_obj.eigenvalues
        assert tica_obj._estimated

        for traj in reader_output[2:]:
            tica_obj.partial_fit(traj)

        # reference
        ref = tica(reader, **params)

        np.testing.assert_allclose(tica_obj.cov, ref.cov, atol=1e-15)
        np.testing.assert_allclose(tica_obj.cov_tau, ref.cov_tau, atol=1e-15)

        np.testing.assert_allclose(tica_obj.eigenvalues, ref.eigenvalues, atol=1e-15)
        # we do not test eigenvectors here, since the system is very metastable and
        # we have multiple eigenvalues very close to one.

if __name__ == "__main__":
    unittest.main()
