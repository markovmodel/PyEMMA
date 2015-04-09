'''
Created on 07.04.2015

@author: marscher
'''
import unittest
import os
import tempfile

import numpy as np
from pyemma.coordinates.io.file_reader import NumPyFileReader
from pyemma.util.log import getLogger


class TestFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.logger = getLogger(cls.__class__.__name__)

        d = np.random.random((100, 3))
        d_1d = np.random.random(100)

        f1 = tempfile.mktemp(suffix='.npy')
        f2 = tempfile.mktemp(suffix='.npy')
        f3 = tempfile.mktemp(suffix='.npz')

        # 2d
        np.save(f1, d)

        # 1d
        np.save(f2, d_1d)

        np.savez(f3, d, d)

        cls.files2d = [f1, f3]
        cls.files1d = [f2]
        cls.d = d
        cls.d_1d = d_1d

        cls.npy_files = [f for f in cls.files2d if f.endswith('.npy')]
        cls.npz = f3

        return cls

    @classmethod
    def tearDownClass(cls):
        # try to clean temporary files
        try:
            for f in cls.files1d:
                os.remove(f)
            for f in cls.files2d:
                os.remove(f)

            os.remove(cls.npz)
        except:
            pass

    def test_only_npy(self):
        reader = NumPyFileReader(self.npy_files)

        from_files = [np.load(f) for f in self.npy_files]
        concatenated = np.vstack(from_files)

        output = reader.get_output()

        self.assertEqual(reader.number_of_trajectories(), len(self.npy_files))
        self.assertEqual(reader.n_frames_total(), concatenated.shape[0])

        for x, y in zip(output, from_files):
            np.testing.assert_equal(x, y)
            
    def test_small_chunks(self):
        reader = NumPyFileReader(self.npy_files)
        reader.chunksize=30

        from_files = [np.load(f) for f in self.npy_files]
        concatenated = np.vstack(from_files)

        output = reader.get_output()

        self.assertEqual(reader.number_of_trajectories(), len(self.npy_files))
        self.assertEqual(reader.n_frames_total(), concatenated.shape[0])

        for x, y in zip(output, from_files):
            np.testing.assert_equal(x, y)

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

        for output, input in zip(all_data, data):
            np.testing.assert_equal(output, input)


if __name__ == "__main__":
    unittest.main()
