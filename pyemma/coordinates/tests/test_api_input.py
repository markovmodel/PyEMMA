import unittest
import os

from pyemma.coordinates.data import MDFeaturizer
from pyemma.util.log import getLogger
import pyemma.coordinates.api as api


logger = getLogger('TestReaderUtils')


class TestApiInput(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.split(__file__)[0], 'data')
        self.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        self.traj_files = [
            os.path.join(path, 'bpti_001-033.xtc'),
            os.path.join(path, 'bpti_067-100.xtc')
        ]

    def tearDown(self):
        pass

    def test_read_multiple_files_topology_file(self):
        reader = api.input(self.traj_files, topology=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_multiple_files_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.input(self.traj_files, featurizer=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_toplogy_file(self):
        reader = api.input(self.traj_files[0], topology=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.input(self.traj_files[0], featurizer=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_invalid_input(self):
        # neither featurizer nor topology file given
        self.assertRaises(ValueError, api.input, self.traj_files, None, None)
        # no input files but a topology file
        self.assertRaises(ValueError, api.input, None, None, self.pdb_file)
        featurizer = MDFeaturizer(self.pdb_file)
        # no input files but a featurizer
        self.assertRaises(ValueError, api.input, None, featurizer, None)
        # empty list of input files
        self.assertRaises(ValueError, api.input, [], None, self.pdb_file)
        # empty tuple of input files
        self.assertRaises(ValueError, api.input, (), None, self.pdb_file)

    def test_invalid_files(self):
        # files do not have the same extension
        self.assertRaises(ValueError, api.input, self.traj_files.append(self.pdb_file), None, self.pdb_file)
        # files list contains something else than strings
        self.assertRaises(ValueError, api.input, self.traj_files.append([2]), None, self.pdb_file)
        # input file is directory
        root_dir = os.path.abspath(os.sep)
        self.assertRaises(ValueError, api.input, root_dir, None, self.pdb_file)


if __name__ == "__main__":
    unittest.main()
