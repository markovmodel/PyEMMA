'''
Created on 07.04.2015

@author: marscher
'''
import unittest
import os
import tempfile

import numpy as np
from pyemma.coordinates.io.file_reader import FileReader, CSVReader


@unittest.skip("jo")
class TestFileReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = np.random.random((100, 3))
        d_1d = np.random.random(100)

        f1 = tempfile.mktemp()
        f2 = tempfile.mktemp(suffix='.npy')
        f3 = tempfile.mktemp()
        f4 = tempfile.mktemp(suffix='.npy')
        f5 = tempfile.mktemp(suffix='.dat')

        npz = tempfile.mktemp(suffix='.npz')

        # 2d
        np.savetxt(f1, d)
        np.save(f2, d)
        np.savetxt(f5, d)

        # 1d
        np.savetxt(f3, d_1d)
        np.save(f4, d_1d)

        np.savez(npz, d, d)

        cls.files2d = [f1, f2, f5]
        cls.files1d = [f3, f4]
        cls.d = d
        cls.d_1d = d_1d

        cls.npz = npz
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
        reader = FileReader(self.files)

        reader.get_output()

    def testSingleFile(self):
        reader = FileReader(self.files2d[0])

        self.assertEqual(reader.n_frames_total(), self.d.shape[0])

    def test_npz(self):
        reader = FileReader(self.npz)
        self.assertEqual(reader.number_of_trajectories(), 2)

    def test_file_1d(self):
        FileReader(self.files1d)

    def test_file_2d(self):
        FileReader(self.files2d)


class TestCSVReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='pyemma_filereader')
        cls.data = np.random.random((10000, 42))
        cls.filename1 = os.path.join(cls.dir, "data.dat")
        np.savetxt(cls.filename1, cls.data)
        return cls

    def test_read(self):
        reader = CSVReader(self.filename1, chunksize=30)

        reader.get_output()


if __name__ == "__main__":
    unittest.main()
