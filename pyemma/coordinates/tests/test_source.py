
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



from __future__ import absolute_import
import unittest
import os
import numpy as np

from pyemma.coordinates.data import MDFeaturizer
from logging import getLogger
import pyemma.coordinates.api as api
import pyemma.util.types as types
import pkg_resources

logger = getLogger('pyemma.'+'TestReaderUtils')


class TestSource(unittest.TestCase):
    def setUp(self):
        path = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data') + os.path.sep
        self.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        self.traj_files = [
            os.path.join(path, 'bpti_001-033.xtc'),
            os.path.join(path, 'bpti_067-100.xtc')
        ]

    def tearDown(self):
        pass

    def test_read_multiple_files_topology_file(self):
        reader = api.source(self.traj_files, top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_multiple_files_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files, features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_toplogy_file(self):
        reader = api.source(self.traj_files[0], top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files[0], features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_invalid_input(self):
        # neither featurizer nor topology file given
        self.assertRaises(ValueError, api.source, self.traj_files, None, None)
        # no input files but a topology file
        self.assertRaises(ValueError, api.source, None, None, self.pdb_file)
        featurizer = MDFeaturizer(self.pdb_file)
        # no input files but a featurizer
        self.assertRaises(ValueError, api.source, None, featurizer, None)
        # empty list of input files
        self.assertRaises(ValueError, api.source, [], None, self.pdb_file)
        # empty tuple of input files
        self.assertRaises(ValueError, api.source, (), None, self.pdb_file)

    def test_invalid_files(self):
        # files do not have the same extension
        self.assertRaises(ValueError, api.source, self.traj_files.append(self.pdb_file), None, self.pdb_file)
        # files list contains something else than strings
        self.assertRaises(ValueError, api.source, self.traj_files.append([2]), None, self.pdb_file)
        # input file is directory
        root_dir = os.path.abspath(os.sep)
        self.assertRaises(ValueError, api.source, root_dir, None, self.pdb_file)

class TestSourceCallAll(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        cls.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        cls.xtc_file = os.path.join(path, 'bpti_mini.xtc')
        cls.inp = api.source(cls.xtc_file, top=cls.pdb_file)

    def setUp(self):
        pass

    def test_chunksize(self):
        assert types.is_int(self.inp.chunksize)

    def test_describe(self):
        desc = self.inp.describe()
        assert types.is_string(desc) or types.is_list_of_string(desc)

    def test_dimension(self):
        assert types.is_int(self.inp.dimension())

    def test_featurizer(self):
        # must have a featurizer
        assert self.inp.featurizer is not None

    def test_get_output(self):
        O = self.inp.get_output()
        assert types.is_list(O)
        assert len(O) == 1
        assert types.is_float_matrix(O[0])
        assert O[0].shape[0] == 100
        assert O[0].shape[1] == self.inp.dimension()

    def test_in_memory(self):
        assert isinstance(self.inp.in_memory, bool)

    def test_iterator(self):
        for itraj, chunk in self.inp:
            assert types.is_int(itraj)
            assert types.is_float_matrix(chunk)
            assert chunk.shape[0] == self.inp.chunksize
            assert chunk.shape[1] == self.inp.dimension()

    def test_n_frames_total(self):
        # map not defined for source
        self.inp.n_frames_total() == 100

    def test_number_of_trajectories(self):
        # map not defined for source
        self.inp.number_of_trajectories() == 1

    def test_output_type(self):
        assert self.inp.output_type() == np.float32

    def test_topfile(self):
        types.is_string(self.inp.topfile)

    def test_trajectory_length(self):
        assert self.inp.trajectory_length(0) == 100
        with self.assertRaises(IndexError):
            self.inp.trajectory_length(1)

    def test_trajectory_lengths(self):
        assert len(self.inp.trajectory_lengths()) == 1
        assert self.inp.trajectory_lengths()[0] == self.inp.trajectory_length(0)

    def test_trajfiles(self):
        assert types.is_list_of_string(self.inp.filenames)

if __name__ == "__main__":
    unittest.main()