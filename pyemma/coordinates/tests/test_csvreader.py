'''
Created on 09.04.2015

@author: marscher
'''
import numpy as np

import unittest
import tempfile
import os

from pyemma.coordinates.io.file_reader import CSVReader
from pyemma.coordinates.io.data_in_memory import DataInMemory


class TestCSVReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='pyemma_filereader')
        cls.nt = 300
        cls.nd = 4
        cls.data = np.arange(cls.nt*cls.nd).reshape(cls.nt, cls.nd)
        cls.filename1 = os.path.join(cls.dir, "data.dat")
        np.savetxt(cls.filename1, cls.data)
        return cls

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.filename1)
        except EnvironmentError:
            pass

    def test_read_1file(self):
        reader = CSVReader(self.filename1, chunksize=4000)

        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.dimension(), self.nd)
        self.assertEqual(reader.n_frames_total(), self.nt)

        # for attr in dir(reader):
        #    print attr, getattr(reader, attr)

        output = reader.get_output()

        np.testing.assert_equal(output[0], self.data)
