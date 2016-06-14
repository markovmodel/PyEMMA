
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
Test feature reader and Tica with a set of cosine time series.
@author: Fabian Paul
'''

from __future__ import print_function

from __future__ import absolute_import
import unittest
import os
import tempfile
import numpy as np
import mdtraj
from pyemma.coordinates import api
from pyemma.coordinates.data.feature_reader import FeatureReader
from logging import getLogger
from six.moves import range

log = getLogger('pyemma.'+'TestFeatureReaderAndTICA')


class TestFeatureReaderAndTICA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 9 # dimension (must be divisible by 3)
        N = 50000  # length of single trajectory # 500000
        N_trajs = 10  # number of trajectories

        cls.w = 2.0*np.pi*1000.0/N  # have 1000 cycles in each trajectory

        # get random amplitudes and phases
        cls.A = np.random.randn(cls.dim)
        cls.phi = np.random.random_sample((cls.dim,))*np.pi*2.0
        mean = np.random.randn(cls.dim)

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb, 'w') as f:
            for i in range(cls.dim//3):
                print(('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i), file=f)

        t = np.arange(0, N)
        t_total = 0
        cls.trajnames = []  # list of xtc file names
        for i in range(N_trajs):
            # set up data
            data = cls.A*np.cos((cls.w*(t+t_total))[:, np.newaxis]+cls.phi) + mean
            xyz = data.reshape((N, cls.dim//3, 3))
            # create trajectory file
            traj = mdtraj.load(cls.temppdb)
            traj.xyz = xyz
            traj.time = t
            tempfname = tempfile.mktemp('.xtc')
            traj.save(tempfname)
            cls.trajnames.append(tempfname)
            t_total += N

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestFeatureReaderAndTICA, cls).tearDownClass()

    def test_covariances_and_eigenvalues(self):
        reader = FeatureReader(self.trajnames, self.temppdb, chunksize=10000)
        for lag in [1, 11, 101, 1001, 2001]:  # avoid cos(w*tau)==0
            trans = api.tica(data=reader, dim=self.dim, lag=lag)
            log.info('number of trajectories reported by tica %d' % trans.number_of_trajectories())
            log.info('tau = %d corresponds to a number of %f cycles' % (lag, self.w*lag/(2.0*np.pi)))
            trans.parametrize()

            # analytical solution for C_ij(lag) is 0.5*A[i]*A[j]*cos(phi[i]-phi[j])*cos(w*lag)
            ana_cov = 0.5*self.A[:, np.newaxis]*self.A*np.cos(self.phi[:, np.newaxis]-self.phi)
            ana_cov_tau = ana_cov*np.cos(self.w*lag)

            self.assertTrue(np.allclose(ana_cov, trans.cov, atol=1.E-3))
            self.assertTrue(np.allclose(ana_cov_tau, trans.cov_tau, atol=1.E-3))
            log.info('max. eigenvalue: %f' % np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))

    def test_partial_fit(self):
        reader = FeatureReader(self.trajnames, self.temppdb, chunksize=10000)
        output = reader.get_output()
        params = {'dim': self.dim, 'lag': 1001}
        ref = api.tica(reader, **params)
        partial = api.tica(**params)

        for traj in output:
            partial.partial_fit(traj)

        np.testing.assert_allclose(partial.eigenvalues, ref.eigenvalues)
        # only compare first two eigenvectors, because we only have two metastable processes
        np.testing.assert_allclose(np.abs(partial.eigenvectors[:2]),
                                   np.abs(ref.eigenvectors[:2]), rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    unittest.main()
