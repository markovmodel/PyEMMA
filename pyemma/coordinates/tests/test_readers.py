from __future__ import absolute_import

import unittest
import tempfile
import shutil
import itertools

import numpy as np

import pyemma.coordinates as coor

import pyemma.coordinates.tests.util as util


class TestReaders(unittest.TestCase):
    """
    trajectory lengths:
        - 5000
        - 6000
        - 1000

    to cover:
    - optimized lagged iterator
    - legacy lagged iterator
    - io-optimization in patches on and off

    - file formats:
        - csv (PyCSVReader)
        - npy (NumpyFileReader)
        - in-memory (DataInMemory)
        - mdtraj-supported formats (FeatureReader)

    - chunk sizes:
        - 0 (whole traj)
        - None (whole traj)
        - 1 (very small chunk size)
        - 100 (medium chunk size)
        - 10000000 (chunk size exceeding file frames)

    - lag:
        - no lag
        - 1 (very small lag)
        - 100 (moderate lag)
        - 5000 (lag so that it exceeds the file frames for at least a couple trajs)

    - fragmented reader
    - sources merger

    The tests:
        - The lag/no-lag cases can be covered in separate tests as we should test if the lagged data is correct
        - When iterating without lag we just have to check that each chunk is <= chunksize (or even == chunksize) and
          that the data matches
        - separate test cases for fragmented reader and sources merger
    """

    chunk_sizes = (0, None, 1, 5, 10, 100, 10000)
    file_formats = ("csv", "in-memory", "numpy", "xtc")
    file_creators = {
        'csv': util.create_trajectory_csv,
        'in-memory': lambda dirname, data: data,
        'numpy': util.create_trajectory_numpy,
        'xtc': util.create_trajectory_xtc
    }
    tempdir = None
    n_atoms = 6
    n_dims = 6*3

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp("test-api-src")
        cls.traj_data = [np.random.random((5000, cls.n_dims)),
                         np.random.random((6000, cls.n_dims)),
                         np.random.random((1000, cls.n_dims))]
        # trajectory files for the different formats
        cls.test_trajs = {}
        for format in cls.file_formats:
            assert format in cls.file_creators.keys()
            cls.test_trajs[format] = [cls.file_creators[format](cls.tempdir, X) for X in cls.traj_data]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    @util.parameterize_test(itertools.product(
        file_formats, chunk_sizes
    ))
    def test_base_reader(self, format, chunksize):
        reader = coor.source(self.test_trajs[format], chunksize=chunksize)


if __name__ == '__main__':
    unittest.main()
