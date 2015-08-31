
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



from __future__ import print_function

from __future__ import absolute_import
import unittest
import os
import tempfile
import numpy as np
import mdtraj
import pyemma.coordinates as coor
from six.moves import range
from six.moves import zip

class TestStride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dim = 3  # dimension (must be divisible by 3)
        N_trajs = 10  # number of trajectories

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb, 'w') as f:
            for i in range(cls.dim//3):
                print(('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i), file=f)

        cls.trajnames = []  # list of xtc file names
        cls.data = []
        for i in range(N_trajs):
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
        for stride in range(1, 100, 23):
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
        for _ in range(N_trajs):
            N = int(np.random.rand()*1000+10)
            d.append(np.random.randn(N, 10).astype(np.float32))

        # read data
        reader = coor.source(d)

        # compare
        for stride in range(1, 10, 3):
            out_reader = reader.get_output(stride=stride)
            for ref_data, test_data in zip(d, out_reader):
                self.assertTrue(np.all(ref_data[::stride] == test_data))  # here we can test exact equality

    def test_parametrize_with_stride(self):
        for stride in range(1, 100, 23):
            r = coor.source(self.trajnames, top=self.temppdb)
            tau = 5
            try:
                t = coor.tica(r, lag=tau, stride=stride, dim=2, force_eigenvalues_le_one=True)
                # force_eigenvalues_le_one=True enables an internal consistency check in TICA
                t.parametrize(stride=stride)
                self.assertTrue(np.all(t.eigenvalues <= 1.0+1.E-12))
            except RuntimeError:
                assert tau % stride != 0

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestStride, cls).tearDownClass()

if __name__ == "__main__":
    unittest.main()