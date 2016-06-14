
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
import tempfile

from pyemma.coordinates.data import MDFeaturizer
from logging import getLogger
import pyemma.coordinates.api as api
import numpy as np
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
from pyemma.coordinates.data.py_csv_reader import PyCSVReader as CSVReader
import shutil


logger = getLogger('pyemma.'+'TestReaderUtils')


class TestApiSourceFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        data_np = np.random.random((100, 3))
        data_raw = np.arange(300 * 4).reshape(300, 4)

        cls.dir = tempfile.mkdtemp("test-api-src")

        cls.npy = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.npz = tempfile.mktemp(suffix='.npz', dir=cls.dir)
        cls.dat = tempfile.mktemp(suffix='.dat', dir=cls.dir)
        cls.csv = tempfile.mktemp(suffix='.csv', dir=cls.dir)

        cls.bs = tempfile.mktemp(suffix=".bs", dir=cls.dir)

        with open(cls.bs, "w") as fh:
            fh.write("meaningless\n")
            fh.write("this can not be interpreted\n")

        np.save(cls.npy, data_np)
        np.savez(cls.npz, data_np, data_np)
        np.savetxt(cls.dat, data_raw)
        np.savetxt(cls.csv, data_raw)

        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        cls.bpti_pdbfile = os.path.join(path, 'bpti_ca.pdb')
        extensions = ['.xtc', '.binpos', '.dcd', '.h5', '.lh5', '.nc', '.netcdf', '.trr']
        cls.bpti_mini_files = [os.path.join(path, 'bpti_mini%s' % ext) for ext in extensions]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_various_formats_source(self):
        chunksizes = [0, 13]
        X = None
        bpti_mini_previous = None
        for cs in chunksizes:
            for bpti_mini in self.bpti_mini_files:
                Y = api.source(bpti_mini, top=self.bpti_pdbfile).get_output(chunk=cs)
                if X is not None:
                    np.testing.assert_array_almost_equal(X, Y, err_msg='Comparing %s to %s failed for chunksize %s'
                                                                       % (bpti_mini, bpti_mini_previous, cs))
                X = Y
                bpti_mini_previous = bpti_mini

    def test_obtain_numpy_file_reader_npy(self):
        reader = api.source(self.npy)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(
            isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")

    @unittest.skip("npz currently unsupported")
    def test_obtain_numpy_file_reader_npz(self):
        reader = api.source(self.npz)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(
            isinstance(reader, NumPyFileReader), "Should be a NumPyFileReader.")

    def test_obtain_csv_file_reader_dat(self):
        reader = api.source(self.dat)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")

    def test_obtain_csv_file_reader_csv(self):
        reader = api.source(self.csv)
        self.assertIsNotNone(reader, "Reader object should not be none.")
        self.assertTrue(isinstance(reader, CSVReader), "Should be a CSVReader.")

    def test_bullshit_csv(self):
        # this file is not parseable as tabulated float file
        self.assertRaises(ValueError, api.source, self.bs)

import pkg_resources
class TestApiSourceFeatureReader(unittest.TestCase):

    def setUp(self):
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep

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
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_multiple_files_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files, features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_toplogy_file(self):
        reader = api.source(self.traj_files[0], top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.filenames, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files[0], features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
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
        self.assertRaises(ValueError, api.source, self.traj_files.append(
            self.pdb_file), None, self.pdb_file)
        # files list contains something else than strings
        self.assertRaises(
            ValueError, api.source, self.traj_files.append([2]), None, self.pdb_file)
        # input file is directory
        root_dir = os.path.abspath(os.sep)
        self.assertRaises(
            ValueError, api.source, root_dir, None, self.pdb_file)


if __name__ == "__main__":
    unittest.main()