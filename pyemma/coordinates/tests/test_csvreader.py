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
Created on 09.04.2015

@author: marscher
'''

from __future__ import absolute_import
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
        cls.filename2 = os.path.join(cls.dir, "data2.dat")
        np.savetxt(cls.filename1, cls.data)
        np.savetxt(cls.filename2, cls.data)

        cls.file_with_header = tempfile.mktemp(suffix=".dat", dir=cls.dir)
        cls.file_with_header2 = tempfile.mktemp(suffix=".dat", dir=cls.dir)

        np.savetxt(cls.file_with_header, cls.data, header="x y z")
        np.savetxt(cls.file_with_header2, cls.data, header="x y z")

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

    def test_read_1file_oneline(self):
        tiny = np.array([1, 2, 3])
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.dat', delete=False) as f:
            np.savetxt(f, tiny)
            f.close()
            reader = CSVReader(f.name, delimiters=" ")
            np.testing.assert_equal(reader.get_output()[0], np.atleast_2d(tiny).T)

    def test_read_1file_with_header_2lines(self):
        data = np.array([1,2,3])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            np.savetxt(f.name, data, header='x')
            f.close()
            out = CSVReader(f.name, delimiters=' ').get_output()[0]
            np.testing.assert_equal(out, np.atleast_2d(data).T)

    def test_read_1file_with_header(self):
        reader = CSVReader(self.file_with_header)
        self.assertEqual(reader.number_of_trajectories(), 1)
        self.assertEqual(reader.dimension(), self.nd)
        self.assertEqual(reader.n_frames_total(), self.nt)

        output = reader.get_output()

        np.testing.assert_almost_equal(output[0], self.data)

    def test_read_2file_with_header(self):
        reader = CSVReader([self.file_with_header, self.file_with_header2])
        self.assertEqual(reader.number_of_trajectories(), 2)
        self.assertEqual(reader.dimension(), self.nd)
        self.assertEqual(reader.n_frames_total(), self.nt*2)

        output = reader.get_output()

        np.testing.assert_almost_equal(output[0], self.data)
        np.testing.assert_almost_equal(output[1], self.data)

    def test_read_with_skipping_first_few_couple_lines(self):
        for skip in [0, 3, 13]:
            r1 = CSVReader(self.filename1, chunksize=30)
            out_with_skip = r1.get_output(skip=skip)[0]
            r2 = CSVReader(self.filename1, chunksize=30)
            out = r2.get_output()[0]
            np.testing.assert_almost_equal(out_with_skip, out[skip::],
                                           err_msg="The first %s rows were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def test_read_with_skipping_first_few_couple_lines_multiple_trajectoryfiles(self):
        for skip in [0, 3, 13]:
            r1 = CSVReader([self.filename1, self.filename2])
            out_with_skip = r1.get_output(skip=skip)
            r2 = CSVReader([self.filename1, self.filename2])
            out = r2.get_output()
            np.testing.assert_almost_equal(out_with_skip[0], out[0][skip::],
                                           err_msg="The first %s rows of the first file were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))
            np.testing.assert_almost_equal(out_with_skip[1], out[1][skip::],
                                           err_msg="The first %s rows of the second file were skipped, but that did not"
                                                   " match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def test_read_lagged_small_chunks(self):
        lag = 200
        reader = CSVReader(self.filename1, chunksize=30)

        lagged_data = self.data[lag:]

        lagged_chunks = []
        it = reader.iterator(lag=lag)
        with it:
            for _, _, Y in it:
                assert len(Y) > 0
                lagged_chunks.append(Y)

        lagged_chunks = np.vstack(lagged_chunks)

        np.testing.assert_almost_equal(lagged_chunks, lagged_data)

    def test_with_kwargs(self):
        args = {'header': 27}

        reader = CSVReader(self.filename1, **args)

        output = reader.get_output()
        np.testing.assert_almost_equal(output[0], self.data)

    def test_with_multiple_files(self):
        files = [self.filename1, self.file_with_header]
        reader = CSVReader(files)

        self.assertEqual(reader.number_of_trajectories(), len(files))

    def test_with_stride(self):
        reader = CSVReader(self.filename1)

        for s in [2, 3, 7, 10]:
            output = reader.get_output(stride=s)[0]
            np.testing.assert_almost_equal(output, self.data[::s], err_msg="stride=%s"%s)

    def test_with_binary_written_file(self):
        data = np.arange(9).reshape(3, 3)
        with tempfile.NamedTemporaryFile('w+b', delete=False) as tmp:
            np.savetxt(tmp.name, data)
            tmp.close()
            out = CSVReader(tmp.name).get_output()[0]
        np.testing.assert_allclose(out, data)

    def test_with_lag(self):
        reader = CSVReader(self.filename1)

        for t in [23, 7, 59]:
            chunks = []
            it = reader.iterator(stride=1, lag=t)
            with it:
                for _, _, Y in it:
                    chunks.append(Y)
            chunks = np.vstack(chunks)
            np.testing.assert_almost_equal(chunks, self.data[t:])

    def test_with_stride_and_lag(self):
        reader = CSVReader(self.filename1)

        for s in [2, 3, 7, 10]:
            for t in [1, 23, 7, 59]:
                chunks = []
                chunks_lag = []
                it = reader.iterator(stride=s, lag=t)
                with it:
                    for _, X, Y in it:
                        chunks.append(X)
                        chunks_lag.append(Y)
                chunks = np.vstack(chunks)
                chunks_lag = np.vstack(chunks_lag)
                actual_lagged = self.data[t::s]
                np.testing.assert_almost_equal(chunks, self.data[::s][0:len(actual_lagged)])
                np.testing.assert_almost_equal(chunks_lag, self.data[t::s],
                                               err_msg="output is not equal for"
                                                       " lag %i and stride %i" % (t, s))

    def test_with_stride_and_lag_with_header(self):
        reader = CSVReader(self.file_with_header)

        for s in [2, 3, 7, 10]:
            for t in [1, 23, 7, 59]:
                chunks = []
                chunks_lag = []
                it = reader.iterator(stride=s, lag=t)
                with it:
                    for _, X, Y in it:
                        chunks.append(X)
                        chunks_lag.append(Y)
                chunks = np.vstack(chunks)
                chunks_lag = np.vstack(chunks_lag)
                actual_lagged = self.data[t::s]
                np.testing.assert_almost_equal(chunks, self.data[::s][0:len(actual_lagged)])
                np.testing.assert_almost_equal(chunks_lag, self.data[t::s],
                                               err_msg="output is not equal for"
                                                       " lag %i and stride %i" % (t, s))

    def test_compare_readline(self):
        data = np.arange(99*3).reshape(-1, 3)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as fh:
            np.savetxt(fh, data)
            # calc offsets
        reader = CSVReader(fh.name)
        assert reader.dimension() == 3
        trajinfo = reader._get_traj_info(fh.name)
        offset = [0]
        with open(fh.name, CSVReader.DEFAULT_OPEN_MODE) as fh2:
            while fh2.readline():
                offset.append(fh2.tell())
            fh2.seek(0)
            lines = []
            for ii, off in enumerate(trajinfo.offsets):
                fh2.seek(off)
                line = fh2.readline()
                fh2.seek(offset[ii])
                line2 = fh2.readline()
                self.assertEqual(line, line2, "differs at offset %i (%s != %s)" % (ii, off, offset[ii]))

if __name__ == '__main__':
    unittest.main()
