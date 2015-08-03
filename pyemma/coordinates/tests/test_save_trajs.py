# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
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


"""
Test the save_trajs function of the coordinates API by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
"""

import unittest
import os
import shutil
import tempfile

import numpy as np
import pyemma.coordinates as coor
from pyemma.coordinates.data.util.reader_utils import single_traj_from_n_files, save_traj_w_md_load_frame, \
    compare_coords_md_trajectory_objects
from pyemma.coordinates.api import save_trajs


class TestSaveTrajs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestSaveTrajs, cls).setUpClass()

    def setUp(self):
        self.eps = 1e-10
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

        self.subdir = tempfile.mkdtemp(suffix='save_trajs_test/')

        # Instantiate the reader
        self.reader = coor.source(self.trajfiles, top=self.pdbfile)
        self.reader.chunksize = 30
        self.n_pass_files = [self.subdir + 'n_pass.set_%06u.xtc' % ii for ii in xrange(len(self.sets))]
        self.one_pass_files = [self.subdir + '1_pass.set_%06u.xtc' % ii for ii in xrange(len(self.sets))]

        self.traj_ref = save_traj_w_md_load_frame(self.reader, self.sets)
        self.strides = [2, 3, 5]

    def tearDown(self):
        shutil.rmtree(self.subdir, ignore_errors=True)

    def test_save_SaveTrajs_IO(self):
        # Test that we're saving to disk alright
        flist = save_trajs(self.reader, self.sets, prefix=self.subdir)
        exist = True
        for f in flist:
            exist = exist and os.stat(f)
        self.assertTrue(exist, "Could not write to disk")

    def test_save_SaveTrajs_multipass(self):

        # Without the "inmemory" option, i.e. multipass
        __ = save_trajs(self.reader, self.sets,
                        outfiles=self.n_pass_files)

        # Reload the object to memory
        traj_n_pass = single_traj_from_n_files(self.n_pass_files, top=self.pdbfile)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_n_pass, self.traj_ref, atom=0)

        self.assertFalse(found_diff, errmsg)

    def test_save_SaveTrajs_onepass(self):

        # With the inmemory option = True
        __ = save_trajs(self.reader, self.sets,
                        outfiles=self.one_pass_files, inmemory=True)

        traj_1_pass = single_traj_from_n_files(self.one_pass_files, top=self.pdbfile)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_1_pass, self.traj_ref, atom=0)
        self.assertFalse(found_diff, errmsg)

    def test_save_SaveTrajs_onepass_with_stride(self):
        # With the inmemory option = True

        for stride in self.strides[:]:
            # Since none of the trajfiles have more than 30 frames, the frames have to be re-drawn for every stride
            sets = np.copy(self.sets)
            sets[0][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[0])[0])
            sets[1][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[1])[0])

            __ = save_trajs(self.reader, sets,
                            outfiles=self.one_pass_files, inmemory=True, stride=stride)

            traj_1_pass = single_traj_from_n_files(self.one_pass_files, top=self.pdbfile)

            # Also the reference has to be re-drawn using the stride. For this, we use the re-scale the strided
            # frame-indexes to the unstrided value
            sets[0][:, 1] *= stride
            sets[1][:, 1] *= stride
            traj_ref = save_traj_w_md_load_frame(self.reader, sets)

            # Check for diffs
            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_1_pass, traj_ref, atom=0)
            self.assertFalse(found_diff, errmsg)

    def test_save_SaveTrajs_multipass_with_stride(self):
        # With the inmemory option = True

        for stride in self.strides[:]:
            # Since none of the trajfiles have more than 30 frames, the frames have to be re-drawn for every stride
            sets = np.copy(self.sets)
            sets[0][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[0])[0])
            sets[1][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[1])[0])

            __ = save_trajs(self.reader, sets,
                            outfiles=self.one_pass_files, inmemory=False, stride=stride)

            traj_1_pass = single_traj_from_n_files(self.one_pass_files, top=self.pdbfile)

            # Also the reference has to be re-drawn using the stride. For this, we use the re-scale the strided
            # frame-indexes to the unstrided value
            sets[0][:, 1] *= stride
            sets[1][:, 1] *= stride
            traj_ref = save_traj_w_md_load_frame(self.reader, sets)

            # Check for diffs
            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_1_pass, traj_ref, atom=0)
            self.assertFalse(found_diff, errmsg)

if __name__ == "__main__":
    unittest.main()
