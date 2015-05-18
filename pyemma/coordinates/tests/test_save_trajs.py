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
Test the save_trajs function of the coordinates API by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
'''

import unittest
import os
import shutil
import tempfile

import numpy as np

import pyemma.coordinates as coor
from pyemma.coordinates.data.util.reader_utils import single_traj_from_n_files
from pyemma.coordinates.api import save_trajs


class TestSaveTrajs(unittest.TestCase):
    # tests that we're able to write to disk

    @classmethod
    def setUpClass(cls):
        super(TestSaveTrajs, cls).setUpClass()

    def setUp(self):
        self.eps = 1e-6
        path = os.path.join(os.path.split(__file__)[0], 'data')
        self.pdbfile = os.path.join(path, 'bpti_ca.pdb')
        self.trajfiles = [os.path.join(path, 'bpti_001-033.xtc'),
                          os.path.join(path, 'bpti_034-066.xtc'),
                          os.path.join(path, 'bpti_067-100.xtc')
                          ]

        # Create random sets of files and frames to be retrieved from trajfiles
        n_members_set1 = 10
        n_members_set2 = 20
        set_1 = np.vstack((np.random.permutation([0, 2] * n_members_set1)[:n_members_set1],
                           np.random.randint(32, size=n_members_set1))).T

        set_2 = np.vstack((np.random.permutation([0, 2] * n_members_set2)[:n_members_set2],
                           np.random.randint(32, size=n_members_set2))).T

        self.sets = [set_1, set_2]

        self.subdir = tempfile.mkdtemp(suffix='save_trajs_test')

        # Instantiate the reader
        self.reader = coor.source(self.trajfiles, top=self.pdbfile)
        self.reader.chunksize = 10
        self.n_pass_files = [self.subdir + 'n_pass.set_%06u.xtc' % ii for ii in xrange(len(self.sets))]
        self.one_pass_files = [self.subdir + '1_pass.set_%06u.xtc' % ii for ii in xrange(len(self.sets))]

    def tearDown(self):
        shutil.rmtree(self.subdir, ignore_errors=True)

    def test_save_SaveDtrajs_IO(self):
        # Test that we're saving to disk alright
        flist = save_trajs(self.reader, self.sets, prefix=self.subdir)
        exist = True
        for f in flist:
            exist = exist and os.stat(f)
        self.assertTrue(exist, "Could not write to disk")

    def test_save_SaveDtrajs_precision(self):

        # Test without the "inmemory" option
        __ = save_trajs(self.reader, self.sets,
                        outfiles=self.n_pass_files)

        # Test with the inmemory option = True
        __ = save_trajs(self.reader, self.sets,
                        outfiles=self.one_pass_files, inmemory=True)

        # Reload the objects to memory
        traj_n_pass = single_traj_from_n_files(self.n_pass_files, top=self.pdbfile)
        traj_1_pass = single_traj_from_n_files(self.one_pass_files, top=self.pdbfile)

        R = np.zeros((2, traj_1_pass.n_frames, 3))
        atom_index = 0

        # Artificially mess the the coordinates
        # traj_1_pass.xyz [0, atom_index, 2] +=1e-5

        for ii, traj in enumerate([traj_n_pass, traj_1_pass]):
            R[ii, :] = traj.xyz[:, atom_index]

        # Compare the R-trajectories among themselves
        found_diff = False
        first_diff = None
        errmsg = ''

        for ii, iR in enumerate(R):
            # Norm of the difference vector

            norm_diff = np.sqrt(((iR - R) ** 2).sum(2))

            # Any differences?
            if (norm_diff > self.eps).any():
                first_diff = np.argwhere(norm_diff > self.eps)[0]
                found_diff = True
                errmsg = "Delta R_%u at frame %u: [%2.1e, %2.1e]" % (atom_index, first_diff[1],
                                                                     norm_diff[0, first_diff[1]],
                                                                     norm_diff[1, first_diff[1]])
                errmsg2 = "\nThe position of atom %u differs by > %2.1e for the same frame between trajectories" % (
                atom_index, self.eps)
                errmsg += errmsg2
                break

        self.assertFalse(found_diff, errmsg)

    def test_with_stride(self):
        strides = [1, 2, 3, 5]

        for stride in strides:
            # Test that we're saving to disk alright
            flist = save_trajs(self.reader, self.sets, prefix=self.subdir, stride=stride)
            exist = True
            for f in flist:
                exist = exist and os.stat(f)
            self.assertTrue(exist, "Could not write to disk")


if __name__ == "__main__":
    unittest.main()
