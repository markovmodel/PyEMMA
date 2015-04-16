
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
Created on 09.04.2015

@author: marscher
'''
import numpy as np

import unittest
import tempfile
import os

from pyemma.coordinates.data.py_csv_reader import PyCSVReader as CSVReader
import shutil


class TestCSVReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='pyemma_filereader')
        cls.nt = 300
        cls.nd = 4
        cls.data = np.arange(cls.nt * cls.nd).reshape(cls.nt, cls.nd)
        cls.filename1 = os.path.join(cls.dir, "data.dat")
        np.savetxt(cls.filename1, cls.data)

        cls.file_with_header = tempfile.mktemp(prefix=".dat", dir=cls.dir)
        np.savetxt(cls.file_with_header, cls.data, header="x y z")

        return cls

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_read_1file(self):
        reader = CSVReader(self.filename1, chunksize=30)

        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.dimension(), self.nd)
        self.assertEqual(reader.n_frames_total(), self.nt)

        output = reader.get_output()

        np.testing.assert_almost_equal(output[0], self.data)

    def test_read_1file_with_header(self):
        reader = CSVReader(self.file_with_header)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.dimension(), self.nd)
        self.assertEqual(reader.n_frames_total(), self.nt)

        output = reader.get_output()

        np.testing.assert_almost_equal(output[0], self.data)

    def test_read_lagged_small_chunks(self):
        lag = 200
        reader = CSVReader(self.filename1, chunksize=30)

        lagged_data = self.data[lag:]

        lagged_chunks = []
        for _, _, Y in reader.iterator(lag=lag):
            # kick out empty chunks
            if Y.shape[0] > 0:
                lagged_chunks.append(Y)

        lagged_chunks = np.vstack(lagged_chunks)

        np.testing.assert_almost_equal(lagged_chunks, lagged_data)

    def test_with_kwargs(self):
        args = {'header': 27}

        reader = CSVReader(self.filename1, **args)

        output = reader.get_output()
        np.testing.assert_almost_equal(output[0], self.data)

    def test_with_multiple_files(self):
        files = [self.filename1, self.filename1]
        reader = CSVReader(files)

        self.assertEqual(reader.number_of_trajectories(), len(files))

    def test_with_stride(self):
        reader = CSVReader(self.filename1)

        for s in [2, 3, 7, 10]:
            output = reader.get_output(stride=s)[0]
            np.testing.assert_almost_equal(output, self.data[::s])

    def test_with_lag(self):
        reader = CSVReader(self.filename1)

        for t in [23, 7, 59]:
            chunks = []
            for _, _, Y in reader.iterator(stride=1, lag=t):
                chunks.append(Y)
            chunks = np.vstack(chunks)
            np.testing.assert_almost_equal(chunks, self.data[t:])

    def test_with_stride_and_lag(self):
        reader = CSVReader(self.filename1)

        for s in [2, 3, 7, 10]:
            for t in [1, 23, 7, 59]:
                chunks = []
                chunks_lag = []
                for _, X, Y in reader.iterator(stride=s, lag=t):
                    chunks.append(X)
                    chunks_lag.append(Y)
                chunks = np.vstack(chunks)
                chunks_lag = np.vstack(chunks_lag)
                np.testing.assert_almost_equal(chunks, self.data[::s])
                np.testing.assert_almost_equal(chunks_lag, self.data[t::s],
                                               err_msg="output is not equal for"
                                               " lag %i and stride %i" % (t, s))

    def test_with_stride_and_lag_with_header(self):
        reader = CSVReader(self.file_with_header)

        for s in [2, 3, 7, 10]:
            for t in [1, 23, 7, 59]:
                chunks = []
                chunks_lag = []
                for _, X, Y in reader.iterator(stride=s, lag=t):
                    chunks.append(X)
                    chunks_lag.append(Y)
                chunks = np.vstack(chunks)
                chunks_lag = np.vstack(chunks_lag)
                np.testing.assert_almost_equal(chunks, self.data[::s])
                np.testing.assert_almost_equal(chunks_lag, self.data[t::s],
                                               err_msg="output is not equal for"
                                               " lag %i and stride %i" % (t, s))
if __name__ == '__main__':
    unittest.main()