
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

import unittest
import os
import tempfile

from pyemma.coordinates.data import MDFeaturizer
from pyemma.util.log import getLogger
import pyemma.coordinates.api as api
import numpy as np
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
from pyemma.coordinates.data.py_csv_reader import PyCSVReader as CSVReader
import shutil


logger = getLogger('TestReaderUtils')


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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

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
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_multiple_files_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files, features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, self.traj_files, "Reader trajectories and input"
                                                                " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_toplogy_file(self):
        reader = api.source(self.traj_files[0], top=self.pdb_file)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
        self.assertListEqual(reader.trajfiles, [self.traj_files[0]], "Reader trajectories and input"
                                                                     " trajectories should coincide.")
        self.assertEqual(reader.featurizer.topologyfile, self.pdb_file, "Featurizers topology file and input "
                                                                        "topology file should coincide.")

    def test_read_single_file_featurizer(self):
        featurizer = MDFeaturizer(self.pdb_file)
        reader = api.source(self.traj_files[0], features=featurizer)
        self.assertIsNotNone(reader, "The reader should not be none.")
        self.assertEqual(reader.topfile, self.pdb_file,
                         "Reader topology file and input topology file should coincide.")
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