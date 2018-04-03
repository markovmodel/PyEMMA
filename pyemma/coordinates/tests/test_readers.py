from __future__ import absolute_import

import unittest
import tempfile
import shutil
import itertools

import numpy as np

import pyemma.coordinates as coor
import pyemma.coordinates.tests.util as util

from pyemma.coordinates.data.feature_reader import FeatureReader


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = funcarglist[0]
    metafunc.parametrize(tuple(argnames.keys()), [[funcargs[name] for name in argnames]
                                    for funcargs in funcarglist])


class TestReaders(object):
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

    tempdir = None
    n_atoms = 6
    n_dims = 6 * 3

    chunk_sizes = (0, 1, 5, 10, 100, 10000)
    strides = (1, 3, 10, 100)
    skips = (0, 123)
    file_formats = ("in-memory", "numpy", "xtc", "trr", "h5")

    # pytest config
    params = {
        'test_base_reader': [dict(file_format=f, stride=s, skip=skip, chunksize=cs)
                             for f, s, skip, cs in itertools.product(file_formats, strides, skips, chunk_sizes)],
    }

    @classmethod
    def setup_class(cls):
        cls.file_creators = {
            'csv': util.create_trajectory_csv,
            'in-memory': lambda dirname, data: data,
            'numpy': util.create_trajectory_numpy,
            'xtc': lambda *args: util.create_trajectory_xtc(cls.n_atoms, *args),
            'trr': lambda *args: util.create_trajectory_trr(cls.n_atoms, *args),
            'h5': lambda *args: util.create_trajectory_h5(cls.n_atoms, *args)
        }
        cls.tempdir = tempfile.mkdtemp("test-api-src")
        cls.traj_data = [np.random.random((5000, cls.n_dims)),
                         np.random.random((6000, cls.n_dims)),
                         np.random.random((1000, cls.n_dims))]
        # trajectory files for the different formats
        cls.test_trajs = {}
        for file_format in cls.file_formats:
            assert file_format in cls.file_creators.keys()
            cls.test_trajs[file_format] = [cls.file_creators[file_format](cls.tempdir, X) for X in cls.traj_data]

        cls.pdb_file = util.create_dummy_pdb(cls.tempdir, cls.n_dims // 3)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def test_base_reader(self, file_format, stride, skip, chunksize):
        trajs = self.test_trajs[file_format]

        if FeatureReader.supports_format(trajs[0]):
            # we need the topology
            reader = coor.source(trajs, top=self.pdb_file, chunksize=chunksize)
        else:
            # no topology required
            reader = coor.source(trajs, chunksize=chunksize)

        np.testing.assert_equal(reader.chunksize, chunksize)

        it = reader.iterator(stride=stride, skip=skip, lag=0, chunk=chunksize)

        assert it.chunksize is not None

        traj_data = [data[skip::stride] for data in self.traj_data]

        with it:
            current_itraj = None
            t = 0
            for itraj, chunk in it:
                # reset t upon next trajectory
                if itraj != current_itraj:
                    current_itraj = itraj
                    t = 0

                assert chunk.shape[0] <= chunksize or chunksize == 0
                if chunksize != 0 and traj_data[itraj].shape[0] - t >= chunksize:
                    assert chunk.shape[0] == chunksize
                elif chunksize == 0:
                    assert chunk.shape[0] == traj_data[itraj].shape[0]

                np.testing.assert_allclose(chunk, traj_data[itraj][t:t+chunk.shape[0]])

                t += chunk.shape[0]


if __name__ == '__main__':
    unittest.main()
