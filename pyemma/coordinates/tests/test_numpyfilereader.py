'''
Created on 07.04.2015

@author: marscher
'''
import unittest
import tempfile

import numpy as np
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
from pyemma.util.log import getLogger
import shutil


class TestFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.logger = getLogger(cls.__class__.__name__)

        d = np.arange(3 * 100).reshape((100, 3))
        d_1d = np.random.random(100)

        cls.dir = tempfile.mkdtemp(prefix='pyemma_npyreader')

        cls.f1 = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.f2 = tempfile.mktemp(suffix='.npy', dir=cls.dir)
        cls.f3 = tempfile.mktemp(suffix='.npz', dir=cls.dir)

        # 2d
        np.save(cls.f1, d)

        # 1d
        np.save(cls.f2, d_1d)

        np.savez(cls.f3, d, d)

        cls.files2d = [cls.f1, cls.f3]
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

    def test_npz(self):
        reader = NumPyFileReader(self.npz)

        all_data = reader.get_output()

        fh = np.load(self.npz)
        data = [x[1] for x in fh.items()]
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

if __name__ == "__main__":
    unittest.main()
