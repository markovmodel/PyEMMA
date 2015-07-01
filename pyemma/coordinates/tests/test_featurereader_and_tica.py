
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
Test feature reader and Tica with a set of cosine time series.
@author: Fabian Paul
'''
import unittest
import os
import tempfile
import numpy as np
import mdtraj
from pyemma.coordinates import api
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.util.log import getLogger

log = getLogger('TestFeatureReaderAndTICA')


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
            for i in xrange(cls.dim//3):
                print>>f, ('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i)

        t = np.arange(0, N)
        t_total = 0
        cls.trajnames = []  # list of xtc file names
        for i in xrange(N_trajs):
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
        trans = api.tica(data=reader, dim=self.dim, lag=1)
        #TICA(tau=1, output_dimension=self.dim)
        for lag in [1, 11, 101, 1001, 2001]:  # avoid cos(w*tau)==0
            log.info('number of trajectories reported by tica %d' % trans.number_of_trajectories())
            log.info('tau = %d corresponds to a number of %f cycles' % (lag, self.w*lag/(2.0*np.pi)))
            trans.lag = lag
            trans.parametrize()

            # analytical solution for C_ij(lag) is 0.5*A[i]*A[j]*cos(phi[i]-phi[j])*cos(w*lag)
            ana_cov = 0.5*self.A[:, np.newaxis]*self.A*np.cos(self.phi[:, np.newaxis]-self.phi)
            ana_cov_tau = ana_cov*np.cos(self.w*lag)

            self.assertTrue(np.allclose(ana_cov, trans.cov, atol=1.E-3))
            self.assertTrue(np.allclose(ana_cov_tau, trans.cov_tau, atol=1.E-3))
            log.info('max. eigenvalue: %f' % np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))

if __name__ == "__main__":
    unittest.main()
