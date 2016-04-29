
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


"""
Test the save_trajs function of the coordinates API by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
"""

from __future__ import absolute_import

import unittest
import os
import shutil
import tempfile

import numpy as np
import pyemma.coordinates as coor
from pyemma.coordinates.data.util.reader_utils import single_traj_from_n_files, save_traj_w_md_load_frame, \
    compare_coords_md_trajectory_objects
from pyemma.coordinates.api import save_trajs
from six.moves import range
import pkg_resources


class TestSaveTrajs(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-10
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
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
        self.n_pass_files = [self.subdir + 'n_pass.set_%06u.xtc' % ii for ii in range(len(self.sets))]
        self.one_pass_files = [self.subdir + '1_pass.set_%06u.xtc' % ii for ii in range(len(self.sets))]

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

    def test_out_of_bound_indexes(self):
        # assert ValueError with index info is raised for faulty input
        self.sets[0][:,1] *= 100000
        with self.assertRaises(ValueError) as raised:
            save_trajs(self.reader, self.sets, outfiles=self.one_pass_files)

if __name__ == "__main__":
    unittest.main()