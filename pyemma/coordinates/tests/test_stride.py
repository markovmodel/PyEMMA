
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

import unittest
import os
import tempfile
import numpy as np
import mdtraj
import pyemma.coordinates as coor

class TestStride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 3  # dimension (must be divisible by 3)
        N_trajs = 10  # number of trajectories

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb, 'w') as f:
            for i in xrange(cls.dim//3):
                print>>f, ('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i)

        cls.trajnames = []  # list of xtc file names
        cls.data = []
        for i in xrange(N_trajs):
            # set up data
            N = int(np.random.rand()*1000+1000)
            xyz = np.random.randn(N, cls.dim//3, 3).astype(np.float32)
            cls.data.append(xyz)
            t = np.arange(0, N)
            # create trajectory file
            traj = mdtraj.load(cls.temppdb)
            traj.xyz = xyz
            traj.time = t
            tempfname = tempfile.mktemp('.xtc')
            traj.save(tempfname)
            cls.trajnames.append(tempfname)

    def test_length_and_content_feature_reader_and_TICA(self):
        for stride in xrange(1, 100, 23):
            r = coor.source(self.trajnames, top=self.temppdb)
            t = coor.tica(data=r, lag=2, dim=2, force_eigenvalues_le_one=True)
            # t.data_producer = r
            t.parametrize()

            # subsample data
            out_tica = t.get_output(stride=stride)
            out_reader = r.get_output(stride=stride)

            # get length in different ways
            len_tica = [x.shape[0] for x in out_tica]
            len_reader = [x.shape[0] for x in out_reader]
            len_trajs = t.trajectory_lengths(stride=stride)
            len_ref = [(x.shape[0]-1)//stride+1 for x in self.data]
            # print 'len_ref', len_ref

            # compare length
            np.testing.assert_equal(len_trajs, len_ref)
            self.assertTrue(len_ref == len_tica)
            self.assertTrue(len_ref == len_reader)

            # compare content (reader)
            for ref_data, test_data in zip(self.data, out_reader):
                ref_data_reshaped = ref_data.reshape((ref_data.shape[0], ref_data.shape[1]*3))
                self.assertTrue(np.allclose(ref_data_reshaped[::stride, :], test_data, atol=1E-3))

    def test_content_data_in_memory(self):
        # prepare test data
        N_trajs = 10
        d = []
        for _ in xrange(N_trajs):
            N = int(np.random.rand()*1000+10)
            d.append(np.random.randn(N, 10).astype(np.float32))

        # read data
        reader = coor.source(d)

        # compare
        for stride in xrange(1, 10, 3):
            out_reader = reader.get_output(stride=stride)
            for ref_data, test_data in zip(d, out_reader):
                self.assertTrue(np.all(ref_data[::stride] == test_data))  # here we can test exact equality

    def test_parametrize_with_stride(self):
        for stride in xrange(1, 100, 23):
            r = coor.source(self.trajnames, top=self.temppdb)
            tau = 5
            t = coor.tica(r, lag=tau, dim=2, force_eigenvalues_le_one=True)
            # force_eigenvalues_le_one=True enables an internal consitency check in TICA
            t.parametrize(stride=stride)
            self.assertTrue(np.all(t.eigenvalues <= 1.0+1.E-12))

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestStride, cls).tearDownClass()

if __name__ == "__main__":
    unittest.main()
