import unittest
import os
import tempfile

from pyemma.coordinates.data import MDFeaturizer
from pyemma.util.log import getLogger
import pyemma.coordinates.api as api
import numpy as np
from pyemma.coordinates.data.file_reader import NumPyFileReader, CSVReader


logger = getLogger('TestReaderUtils')


class TestApiSourceFileReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_np = np.random.random((100, 3))
        data_raw = np.arange(300 * 4).reshape(300, 4)

        cls.npy = tempfile.mktemp(suffix='.npy')
        cls.npz = tempfile.mktemp(suffix='.npz')
        cls.dat = tempfile.mktemp(suffix='.dat')
        cls.csv = tempfile.mktemp(suffix='.csv')

        np.save(cls.npy, data_np)
        np.savez(cls.npz, data_np, data_np)
        np.savetxt(cls.dat, data_raw)
        np.savetxt(cls.csv, data_raw)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.npy)
        os.remove(cls.npz)
        os.remove(cls.dat)
        os.remove(cls.csv)

    # def test_obtain_numpy_file_reader_npy(self):
    #     reader = api.source(self.npy)
    #     self.assertIsNotNone(reader, "Reader object should not be none.")
    #     self.assertTrue(isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")
    #
    # def test_obtain_numpy_file_reader_npz(self):
    #     reader = api.source(self.npz)
    #     self.assertIsNotNone(reader, "Reader object should not be none.")
    #     self.assertTrue(isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")
    #
    # def test_obtain_csv_file_reader_dat(self):
    #     reader = api.source(self.dat)
    #     self.assertIsNotNone(reader, "Reader object should not be none.")
    #     self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")
    #
    # def test_obtain_csv_file_reader_csv(self):
    #     reader = api.source(self.csv)
    #     self.assertIsNotNone(reader, "Reader object should not be none.")
    #     self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")


class TestApiSourceFeatureReader(unittest.TestCase):
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
        reader = api.source(self.traj_files, top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_multiple_files_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files, features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_toplogy_file(self):
        reader = api.source(self.traj_files[0], top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files[0], features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file, "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, [self.traj_files[0]], "Reader trajectories and input"
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


if __name__ == "__main__":
    unittest.main()