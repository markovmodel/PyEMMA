
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
Created on 07.04.2015

@author: marscher
'''

from __future__ import print_function

from __future__ import absolute_import
import unittest
import tempfile

import numpy as np
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
from pyemma.util.log import getLogger
import shutil
from six.moves import range
from six.moves import zip


class TestNumPyFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.logger = getLogger(cls.__class__.__name__)

        d = np.arange(3 * 100).reshape((100, 3))
        d2 = np.arange(300, 900).reshape((200,3))
        d_1d = np.random.random(100)

        cls.dir = tempfile.mkdtemp(prefix='pyemma_npyreader')

        cls.f1 = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.f2 = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.f3 = tempfile.mktemp(suffix='.npz', dir=cls.dir)
        cls.f4 = tempfile.mktemp(suffix='.npy', dir=cls.dir)


        # 2d
        np.save(cls.f1, d)
        np.save(cls.f4, d2)

        # 1d
        np.save(cls.f2, d_1d)

        np.savez(cls.f3, d, d)

        cls.files2d = [cls.f1, cls.f4] #cls.f3]
        cls.files1d = [cls.f2]
        cls.d = d
        cls.d_1d = d_1d

        cls.npy_files = [f for f in cls.files2d if f.endswith('.npy')]
        cls.npz = cls.f3

        return cls

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_only_npy(self):
        reader = NumPyFileReader(self.npy_files)

        from_files = [np.load(f) for f in self.npy_files]
        concatenated = np.vstack(from_files)

        output = reader.get_output()

        self.assertEqual(reader.number_of_trajectories(), len(self.npy_files))
        self.assertEqual(reader.n_frames_total(), concatenated.shape[0])

        for x, y in zip(output, from_files):
            np.testing.assert_array_almost_equal(x, y)

    def test_small_chunks(self):
        reader = NumPyFileReader(self.npy_files)
        reader.chunksize = 30

        from_files = [np.load(f) for f in self.npy_files]
        concatenated = np.vstack(from_files)

        output = reader.get_output()

        self.assertEqual(reader.number_of_trajectories(), len(self.npy_files))
        self.assertEqual(reader.n_frames_total(), concatenated.shape[0])

        for x, y in zip(output, from_files):
            np.testing.assert_array_almost_equal(x, y)

    def testSingleFile(self):
        reader = NumPyFileReader(self.npy_files[0])

        self.assertEqual(reader.n_frames_total(), self.d.shape[0])

    @unittest.skip("npz currently unsupported")
    def test_npz(self):
        reader = NumPyFileReader(self.npz)

        all_data = reader.get_output()

        fh = np.load(self.npz)
        data = [x[1] for x in list(fh.items())]
        fh.close()

        self.assertEqual(reader.number_of_trajectories(), len(data))

        for outp, inp in zip(all_data, data):
            np.testing.assert_equal(outp, inp)

    def test_stridden_access(self):
        reader = NumPyFileReader(self.f1)
        reader.chunksize = 10

        wanted = np.load(self.f1)

        for stride in [2, 3, 5, 7, 15]:
            first_traj = reader.get_output(stride=stride)[0]
            np.testing.assert_equal(first_traj, wanted[::stride],
                                    "did not match for stride %i" % stride)

    def test_lagged_stridden_access(self):
        reader = NumPyFileReader(self.f1)
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        for stride in strides:
            for lag in lags:
                chunks = []
                for _, _, Y in reader.iterator(stride, lag):
                    chunks.append(Y)
                chunks = np.vstack(chunks)
                np.testing.assert_equal(chunks, self.d[lag::stride])

    def test_lagged_stridden_access_multiple_files(self):
        reader = NumPyFileReader(self.files2d)
        print(reader.trajectory_lengths())
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        for stride in strides:
            for lag in lags:
                chunks = {i: [] for i in range(reader.number_of_trajectories())}
                for itraj, _, Y in reader.iterator(stride, lag):
                    chunks[itraj].append(Y)

                for i, k in enumerate(chunks.values()):
                    stack = np.vstack(k)
                    d = np.load(self.files2d[i])
                    np.testing.assert_equal(stack, d[lag::stride],
                                            "not equal for stride=%i"
                                            " and lag=%i" % (stride, lag))

    def test_lagged_stridden_access_multiple_files_dims(self):
        reader = NumPyFileReader(self.files2d, usecols=[1])
        print(reader.trajectory_lengths())
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        for stride in strides:
            for lag in lags:
                chunks = {i: [] for i in range(reader.number_of_trajectories())}
                for itraj, _, Y in reader.iterator(stride, lag):
                    chunks[itraj].append(Y)

                for i, k in enumerate(chunks.values()):
                    stack = np.vstack(k)
                    d = np.load(self.files2d[i])[:,reader._usecols]
                    np.testing.assert_equal(stack, d[lag::stride],
                                            "not equal for stride=%i"
                                            " and lag=%i" % (stride, lag))

if __name__ == "__main__":
    unittest.main()