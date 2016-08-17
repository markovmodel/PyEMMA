
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
Test the get_frames_from_file by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
'''

from __future__ import absolute_import

import pkg_resources
import unittest
import os

from numpy.random import randint
from numpy import floor, allclose
import mdtraj as md

from pyemma.coordinates.data.util.frames_from_file import frames_from_files as _frames_from_file
from pyemma.coordinates.data.util.reader_utils import compare_coords_md_trajectory_objects


class TestFramesFromFile(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-10
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        self.pdbfile = os.path.join(path, 'bpti_ca.pdb')
        self.trajfiles = os.path.join(path, 'bpti_mini.xtc')

        # Create of frames to be retrieved from trajfiles
        self.n_frames = 50
        self.frames = randint(0, high = 100, size = self.n_frames)
        self.chunksize = 30

        self.mdTrajectory = md.load(self.pdbfile)

    def test_returns_trajectory(self):
        assert isinstance(_frames_from_file(self.trajfiles, self.pdbfile, self.frames),
                          md.Trajectory)

    def test_gets_the_right_frames_no_stride_no_chunk(self):
        # I am calling this "no_chunk" because chunksize = int(1e3) will force frames_from_file to load one single chunk

        traj_test = _frames_from_file(self.trajfiles, self.pdbfile, self.frames, chunksize = int(1e3), verbose=False)
        traj_ref = md.load(self.trajfiles, top = self.pdbfile)[self.frames]

        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
        self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_no_stride_with_chunk(self):

        traj_test = _frames_from_file(self.trajfiles, self.pdbfile, self.frames, chunksize=self.chunksize, verbose = False)
        traj_ref = md.load(self.trajfiles, top=self.pdbfile)[self.frames]

        (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom = 0, mess = False)
        self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_with_stride_no_chunk(self):
        # I am calling this "no_chunk" because chunksize = int(1e3) will force frames_from_file to load one single chunk

        for stride in [2, 5, 10]:
            # Make sure we don't overshoot the number of available frames (100)
            frames = randint(0, high=floor(100 / stride), size=self.n_frames)

            traj_test = _frames_from_file(self.trajfiles, self.pdbfile, frames, stride = stride, verbose=False)
            traj_ref = md.load(self.trajfiles, top=self.pdbfile, stride = stride)[frames]

            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
            self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_with_stride_with_chunk(self):

        for stride in [2, 3, 5, 6, 10, 15]:
            # Make sure we don't overshoot the number of available frames (100)
            frames = randint(0, high = floor(100/stride), size = self.n_frames)

            traj_test = _frames_from_file(self.trajfiles, self.pdbfile, frames,
                                          chunksize=self.chunksize,
                                          stride = stride,
                                          verbose=False)
            traj_ref = md.load(self.trajfiles, top=self.pdbfile, stride = stride)[frames]

            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
            self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_with_stride_with_chunk_mdTrajectory_input(self):

        for stride in [2, 3, 5, 6, 10, 15]:
            # Make sure we don't overshoot the number of available frames (100)
            frames = randint(0, high = floor(100/stride), size = self.n_frames)

            traj_test = _frames_from_file(self.trajfiles, self.mdTrajectory, frames,
                                          chunksize=self.chunksize,
                                          stride = stride,
                                          verbose=False)
            traj_ref = md.load(self.trajfiles, top=self.pdbfile, stride = stride)[frames]

            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
            self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_with_stride_with_chunk_mdTopology_input(self):

        for stride in [2, 3, 5, 6, 10, 15]:
            # Make sure we don't overshoot the number of available frames (100)
            frames = randint(0, high = floor(100/stride), size = self.n_frames)

            traj_test = _frames_from_file(self.trajfiles, self.mdTrajectory.top, frames,
                                          chunksize=self.chunksize,
                                          stride = stride,
                                          verbose=False)
            traj_ref = md.load(self.trajfiles, top=self.pdbfile, stride = stride)[frames]

            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
            self.assertFalse(found_diff, errmsg)

    def test_gets_the_right_frames_with_stride_with_copy(self):

        for stride in [2, 3, 5, 6, 10, 15]:
            # Make sure we don't overshoot the number of available frames (100)
            frames = randint(0, high = floor(100/stride), size = self.n_frames)

            traj_test = _frames_from_file(self.trajfiles, self.pdbfile, frames,
                                          chunksize=self.chunksize,
                                          stride = stride,
                                          verbose=False,
                                          copy_not_join=True
                                          )
            traj_ref = md.load(self.trajfiles, top=self.pdbfile, stride = stride)[frames]

            (found_diff, errmsg) = compare_coords_md_trajectory_objects(traj_test, traj_ref, atom=0, mess=False)
            self.assertFalse(found_diff, errmsg)
            assert allclose(traj_test.unitcell_lengths, traj_ref.unitcell_lengths)
            assert allclose(traj_test.unitcell_angles, traj_ref.unitcell_angles)


if __name__ == "__main__":
    unittest.main()
