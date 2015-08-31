
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