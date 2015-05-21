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
Test the get_frames_from_file by comparing
the direct, sequential retrieval of frames via mdtraj.load_frame() vs
the retrival via save_trajs
@author: gph82, clonker
'''

import unittest
import os

from numpy.random import randint
from numpy import floor
import mdtraj as md
from pyemma.coordinates.data.frames_from_file import frames_from_file as _frames_from_file
from pyemma.coordinates.data.util.reader_utils import compare_coords_md_trajectory_objects

class TestFramesFromFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFramesFromFile, cls).setUpClass()

    def setUp(self):
        self.eps = 1e-10
        path = os.path.join(os.path.split(__file__)[0], 'data')
        self.pdbfile = os.path.join(path, 'bpti_ca.pdb')
        self.trajfiles = os.path.join(path, 'bpti_mini.xtc')

        # Create of frames to be retrieved from trajfiles
        self.n_frames = 50
        self.frames = randint(0, high = 100, size = self.n_frames)
        self.chunksize = 30

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

if __name__ == "__main__":
    unittest.main()
