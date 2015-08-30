
'''
Created on 22.01.2015

@author: marscher
'''

from __future__ import absolute_import
import os
import tempfile
import unittest
import numpy as np

from pyemma.coordinates.data.writer import WriterCSV
from pyemma.coordinates.data.data_in_memory import DataInMemory


class TestWriterCSV(unittest.TestCase):

    def setUp(self):
        self.output_file = tempfile.mktemp('', 'test_writer_csv')

    def tearDown(self):
        os.unlink(self.output_file)

    def testWriter(self):
        writer = WriterCSV(self.output_file)
        data = np.random.random((100, 3))
        dm = DataInMemory(data)
        writer.data_producer = dm

        writer.parametrize()

        # open file and compare data
        output = np.loadtxt(self.output_file)
        np.testing.assert_allclose(output, data)

if __name__ == "__main__":
    unittest.main()