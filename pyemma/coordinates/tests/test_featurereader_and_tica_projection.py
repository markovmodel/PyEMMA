
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
Test feature reader and Tica by checking the properties of the ICs.
cov(ic_i,ic_j) = delta_ij and cov(ic_i,ic_j,tau) = lambda_i delta_ij
@author: Fabian Paul
'''
import unittest
import os
import tempfile
import numpy as np
import mdtraj
from pyemma.coordinates.api import tica, _TICA as TICA
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.util.log import getLogger

log = getLogger('TestFeatureReaderAndTICAProjection')

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
            for i in xrange(cls.dim // 3):
                print>>f, ('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i)

        t = np.arange(0, N)
        cls.trajnames = []  # list of xtc file names
        for i in xrange(N_trajs):
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
        trans = TICA(lag=1, dim=self.dim, force_eigenvalues_le_one=True)
        trans.data_producer = reader
        for tau in [1, 10, 100, 1000, 2000]:
            log.info('number of trajectories reported by tica %d' % trans.number_of_trajectories())
            trans.lag = tau
            trans.parametrize()
            data = trans.get_output()
            # print '@@cov', trans.cov
            # print '@@cov_tau', trans.cov_tau

            log.info('max. eigenvalue: %f' % np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))
            # check ICs
            check = tica(data=data, lag=tau, dim=self.dim, force_eigenvalues_le_one=True)
            check.parametrize()

            self.assertTrue(np.allclose(np.eye(self.dim), check.cov))
            self.assertTrue(np.allclose(check.mu, 0.0))
            ic_cov_tau = np.zeros((self.dim, self.dim))
            ic_cov_tau[np.diag_indices(self.dim)] = trans.eigenvalues
            self.assertTrue(np.allclose(ic_cov_tau, check.cov_tau))
            # print '@@cov_tau', check.cov_tau

if __name__ == "__main__":
    unittest.main()
