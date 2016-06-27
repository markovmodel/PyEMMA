
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
the retrival via save_traj
@author: gph82, clonker
"""

from __future__ import absolute_import

import unittest
import os
import shutil
import tempfile
import pkg_resources

import numpy as np
import pyemma

import pyemma.coordinates as coor
import mdtraj as md
from pyemma.coordinates.data.util.reader_utils import single_traj_from_n_files, save_traj_w_md_load_frame, \
    compare_coords_md_trajectory_objects
from pyemma.coordinates.api import save_traj
from six.moves import range

from pyemma.coordinates.tests.util import create_traj, get_top


class TestSaveTraj(unittest.TestCase):

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
        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(34):
            set_1 = np.vstack((np.random.permutation([0, 2] * n_members_set1)[:n_members_set1],
                               np.random.randint(32, size=n_members_set1))).T

            set_2 = np.vstack((np.random.permutation([0, 2] * n_members_set2)[:n_members_set2],
                               np.random.randint(32, size=n_members_set2))).T

        self.sets = [set_1, set_2]

        self.subdir = tempfile.mkdtemp(suffix='save_trajs_test/')
        self.outfile = os.path.join(self.subdir, 'save_traj_test.xtc')

        # Instantiate the reader
        self.reader = coor.source(self.trajfiles, top=self.pdbfile)
        self.reader.chunksize = 30
        self.n_pass_files = [self.subdir + 'n_pass.set_%06u.xtc' % ii for ii in range(len(self.sets))]
        self.one_pass_files = [self.subdir + '1_pass.set_%06u.xtc' % ii for ii in range(len(self.sets))]

        self.traj_ref = save_traj_w_md_load_frame(self.reader, self.sets)
        self.strides = [2, 3, 5]

    def tearDown(self):
        shutil.rmtree(self.subdir, ignore_errors=True)

    def test_reader_input_save_IO(self):
        # Test that we're saving to disk alright
        save_traj(self.reader, self.sets, self.outfile)
        exist = os.stat(self.outfile)
        self.assertTrue(exist, "Could not write to disk")

    def test_reader_input_returns_trajectory(self):
        self.assertTrue(isinstance(save_traj(self.reader, self.sets, None),
                          md.Trajectory))

    def test_reader_input_returns_trajectory_w_image_molecules(self):
        self.assertTrue(isinstance(save_traj(self.reader, self.sets, None, image_molecules=True),
                          md.Trajectory))

    def test_list_input_save_IO(self):
        # Test that we're saving to disk alright
        save_traj(self.trajfiles, self.sets, self.outfile, top=self.pdbfile)
        exist = os.stat(self.outfile)
        self.assertTrue(exist, "Could not write to disk")

    def test_list_input_returns_trajectory(self):
        self.assertTrue(isinstance(save_traj(self.trajfiles, self.sets, None, top=self.pdbfile),
                          md.Trajectory))

    def test_reader_input_save_correct_frames_disk(self):

        save_traj(self.reader, self.sets, self.outfile)

        # Reload the object to memory
        traj = md.load(self.outfile, top=self.pdbfile)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, self.traj_ref, atom=0)

        self.assertFalse(found_diff, errmsg)

    def test_reader_input_save_correct_frames_mem(self):

        # Keep object in memory
        traj = save_traj(self.reader, self.sets, None)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, self.traj_ref, atom=0)

        self.assertFalse(found_diff, errmsg)

    def test_list_input_save_correct_frames_disk(self):

        save_traj(self.trajfiles, self.sets, self.outfile, top=self.pdbfile)

        # Reload the object to memory
        traj = md.load(self.outfile, top=self.pdbfile)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, self.traj_ref, atom=0)

        self.assertFalse(found_diff, errmsg)

    def test_list_input_save_correct_frames_mem(self):

        # Keep object in memory
        traj = save_traj(self.trajfiles, self.sets, None, top=self.pdbfile)

        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, self.traj_ref, atom=0)

        self.assertFalse(found_diff, errmsg)

    def test_reader_input_save_correct_frames_with_stride_in_memory(self):
        # With the inmemory option = True

        for stride in self.strides[:]:
            # Since none of the trajfiles have more than 30 frames, the frames have to be re-drawn for every stride
            sets = np.copy(self.sets)
            sets[0][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[0])[0])
            sets[1][:, 1] = np.random.randint(0, high=30 / stride, size=np.shape(sets[1])[0])

            traj = save_traj(self.reader, sets, None,
                             stride=stride, verbose=False)

            # Also the reference has to be re-drawn using the stride. For this, we use the re-scale the strided
            # frame-indexes to the unstrided value
            sets[0][:, 1] *= stride
            sets[1][:, 1] *= stride
            traj_ref = save_traj_w_md_load_frame(self.reader, sets)

            # Check for diffs
            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, traj_ref, atom=0)
            self.assertFalse(found_diff, errmsg)

    def test_with_fragmented_reader(self):
        from pyemma.util.files import TemporaryDirectory
        trajlen = 35
        # trajectory 0 (first trajectory, is trajfiles[2])
        #   -> skipped
        # trajectory 1 (second trajectory, is {trajfiles[0], trajfiles[1]})
        #   fragment 1:
        #       -> frames 0,1,2,10
        #   fragment 2:
        #       -> frames 1 (i.e., 36) and 34 (i.e., 69)
        # trajectory 2 (third trajectory, is trajfiles[2])
        #   -> frame 5
        ra_indices = np.array([[1, 0], [1, 1], [1,2], [1, 10], [1, trajlen+1], [1, 2*trajlen-1], [2, 5]], dtype=int)
        with TemporaryDirectory() as td:

            trajfiles = []
            xyzs = []
            for i in range(3):
                tf, xyz, _ = create_traj(start=i * 10, dir=td, length=trajlen)
                trajfiles.append(tf)
                xyzs.append(xyz)

            topfile = get_top()
            frag_traj = [trajfiles[2], [trajfiles[0], trajfiles[1]], trajfiles[2]]

            expected = xyzs[0][np.array([0, 1, 2, 10]), :], xyzs[1][np.array([1, 34])], np.array([(xyzs[2][5, :])])
            expected = np.vstack(expected)

            reader = coor.source(frag_traj, top=topfile)

            for cs in range(1,10):
                traj = save_traj(reader, ra_indices, None, chunksize=cs)
                np.testing.assert_almost_equal(traj.xyz, expected)

    def test_with_fragmented_reader_chunksize_0(self):
        # intentionally group bpti dataset to a fake fragmented traj
        frag_traj = [[self.trajfiles[0], self.trajfiles[1]], self.trajfiles[2], self.trajfiles[2]]
        reader = coor.source(frag_traj, top=self.pdbfile, chunk_size=0)
        assert reader.chunksize == 0
        traj = save_traj(reader, self.sets, None)
        traj_ref = save_traj_w_md_load_frame(self.reader, self.sets)
        # Check for diffs
        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj, traj_ref, atom=0)

        np.testing.assert_equal(traj.xyz, traj_ref.xyz)
        self.assertFalse(found_diff, errmsg)

    def test_invalid_maximum_traj_index(self):
        frag_traj = [[self.trajfiles[0], self.trajfiles[1]], self.trajfiles[2], self.trajfiles[2]]
        set = [[0,2], [0,1], [2,42]]
        from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
        reader = FragmentedTrajectoryReader(frag_traj, topologyfile=self.pdbfile)
        with self.assertRaises(ValueError) as cm:
            save_traj(reader, set, None)
        self.assertIn("larger than", cm.exception.args[0])

    def test_invalid_readers_in_frag_traj(self):
        data = [np.array([[[1,2], [3,4]],[0,1]])]
        from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
        reader = FragmentedTrajectoryReader(data)
        with self.assertRaises(ValueError) as cm:
            save_traj(reader, self.sets, None)
        self.assertIn("FeatureReader", cm.exception.args[0])

if __name__ == "__main__":
    unittest.main()
