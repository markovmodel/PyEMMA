
import unittest
import tempfile
import shutil
import itertools
from collections import defaultdict

import numpy as np

import pyemma.coordinates as coor
import pyemma.coordinates.tests.util as util
from pyemma.coordinates.data import FragmentedTrajectoryReader


def max_chunksize_from_config(itemsize):
    from pyemma import config
    from pyemma.util.units import string_to_bytes
    max_bytes = string_to_bytes(config.default_chunksize)
    max_frames = max(1, int(np.floor(max_bytes / itemsize)))
    return max_frames


class GenerateTestMatrix(type):
    def __new__(mcs, name, bases, attr):
        from functools import partial

        # needed for python2
        class partialmethod(partial):
            def __get__(self, instance, owner):
                if instance is None:
                    return self
                return partial(self.func, instance,
                               *(self.args or ()), **(self.keywords or {}))
        new_test_methods = {}

        test_templates = {k: v for k, v in attr.items() if k.startswith('_test') }
        test_parameters = attr['params']
        for test, params in test_templates.items():
            test_param = test_parameters[test]
            for param_set in test_param:
                func = partialmethod(attr[test], **param_set)
                # only 'primitive' types should be used as part of test name.
                vals_str = '_'.join((str(v) if not isinstance(v, np.ndarray) else 'array' for v in param_set.values()))
                assert '[' not in vals_str, 'this char makes pytest think it has to extract parameters out of the testname.'
                out_name = '{}_{}'.format(test[1:], vals_str)
                func.__qualname__ = 'TestReaders.{}'.format(out_name)
                new_test_methods[out_name] = func

        attr.update(new_test_methods)
        return type.__new__(mcs, name, bases, attr)


class TestReaders(unittest.TestCase, metaclass=GenerateTestMatrix):
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

    chunk_sizes = (0, 1, 127, 5000, 10000, None)
    strides = (1, 3, 100)
    ra_strides = (
        np.array([[0, 1], [0, 3], [0, 3], [0, 5], [0, 6], [0, 7], [2, 1], [2, 1]]),
        np.array([[0, 23], [0, 42], [0, 4999], [1, 3], [2, 666], [3, 42], [4, 128]])
    )
    skips = (0, 123)
    lags = (1, 128, 5000)
    file_formats = ('in-memory',
                    'numpy',
                    'xtc',
                    'trr',
                    'dcd',
                    'h5',  # pyemma H5Reader
                    'csv',
                    )
    # transform data or not (identity does not change the data, but pushes it through a StreamingTransformer).
    transforms = (None, 'identity')

    eps = np.finfo(np.float32).eps

    # pytest config
    params = {
        '_test_base_reader': [dict(file_format=f, stride=s, skip=skip, chunksize=cs, transform=t)
                              for f, s, skip, cs,t  in itertools.product(file_formats, strides, skips,
                                                                         chunk_sizes, transforms)],
        '_test_lagged_reader': [
            dict(file_format=f, stride=s, skip=skip, chunksize=cs, lag=lag)
            for f, s, skip, cs, lag in itertools.product(file_formats, strides, skips, chunk_sizes, lags)
        ],
        '_test_fragment_reader': [
            dict(file_format=f, stride=s, lag=l, chunksize=cs)
            for f, s, l, cs in itertools.product(file_formats, strides, [0] + list(lags), chunk_sizes)
        ],
        '_test_base_reader_with_random_access_stride': [
          dict(file_format=f, stride=s, chunksize=cs)
            for f, s, cs in itertools.product(file_formats, ra_strides, chunk_sizes)
        ]
    }

    @classmethod
    def setup_class(cls):
        cls.file_creators = {
            'csv': util.create_trajectory_csv,
            'in-memory': lambda dirname, data: data,
            'numpy': util.create_trajectory_numpy,
            'xtc': lambda dirname, data: util.create_trajectory_xtc(cls.n_atoms, dirname, data),
            'trr': lambda dirname, data: util.create_trajectory_trr(cls.n_atoms, dirname, data),
            'dcd': lambda dirname, data: util.create_trajectory_dcd(cls.n_atoms, dirname, data),
            'h5': lambda dirname, data: util.create_trajectory_h5(cls.n_atoms, dirname, data)
        }
        cls.tempdir = tempfile.mkdtemp("test-api-src")
        cls.traj_data = [np.random.random((5000, cls.n_dims)).astype(np.float32),
                         np.random.random((50, cls.n_dims)).astype(np.float32),
                         np.random.random((6000, cls.n_dims)).astype(np.float32),
                         np.random.random((125, cls.n_dims)).astype(np.float32),
                         np.random.random((1000, cls.n_dims)).astype(np.float32)]
        # trajectory files for the different formats
        cls.test_trajs = {}
        for file_format in cls.file_formats:
            assert file_format in cls.file_creators.keys()
            cls.test_trajs[file_format] = [cls.file_creators[file_format](cls.tempdir, X) for X in cls.traj_data]

        cls.pdb_file = util.create_dummy_pdb(cls.tempdir, cls.n_dims // 3)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    # H5Reader does not need a topology (and gets only selected in case we do not pass it).
    def setUp(self):
        if 'h5' in self._testMethodName:
            self.pdb_file_bak = self.pdb_file
            self.pdb_file = None

    def tearDown(self):
        if 'h5' in self._testMethodName:
            self.pdb_file = self.pdb_file_bak

    def _test_lagged_reader(self, file_format, stride, skip, chunksize, lag):
        trajs = self.test_trajs[file_format]
        reader = coor.source(trajs, top=self.pdb_file, chunksize=chunksize)

        it = reader.iterator(stride=stride, skip=skip, lag=lag, chunk=chunksize)
        traj_data = [data[skip::stride] for data in self.traj_data]
        traj_data_lagged = [data[skip + lag::stride] for data in self.traj_data]
        valid_itrajs = [i for i, x in enumerate(traj_data_lagged) if len(x) > 0]

        assert it.chunksize is not None
        if chunksize is None:
            chunksize = max_chunksize_from_config(reader.output_type().itemsize)

        with it:
            current_itraj = None
            t = t_total = 0
            collected = defaultdict(list)
            collected_lag = defaultdict(list)

            for itraj, chunk, chunk_lagged in it:
                # reset t upon next trajectory
                if itraj != current_itraj:
                    current_itraj = itraj
                    t = 0
                assert len(chunk) <= chunksize or chunksize == 0
                if chunksize != 0 and len(traj_data[itraj]) - t >= chunksize:
                    assert len(chunk) <= chunksize
                elif chunksize == 0:
                    assert len(chunk) == len(chunk_lagged) == len(traj_data_lagged[itraj])
                collected[itraj].append(chunk)
                collected_lag[itraj].append(chunk_lagged)

                t += len(chunk)
                t_total += len(chunk)

        for itraj in valid_itrajs:
            assert itraj in collected.keys()
            assert itraj in collected_lag.keys()

        assert set(collected.keys()) == set(collected_lag.keys())
        for itraj in collected.keys():
            assert itraj in valid_itrajs
            collected[itraj] = np.vstack(collected[itraj])
            collected_lag[itraj] = np.vstack(collected_lag[itraj])
            # unlagged data is truncated to the length of the lagged data.
            max_len = len(traj_data_lagged[itraj])
            np.testing.assert_allclose(collected[itraj], traj_data[itraj][:max_len], atol=self.eps)
            np.testing.assert_allclose(collected_lag[itraj], traj_data_lagged[itraj], atol=self.eps)

        assert t_total == sum(len(x) for x in collected.values())
        assert t_total == reader.n_frames_total(stride=stride, skip=skip+lag)

    def _test_fragment_reader(self, file_format, stride, lag, chunksize):
        trajs = self.test_trajs[file_format]

        reader = coor.source([trajs], top=self.pdb_file, chunksize=chunksize)
        assert isinstance(reader, FragmentedTrajectoryReader)

        data = np.vstack(self.traj_data)
        itraj = None

        if lag > 0:
            collected = []
            collected_lagged = []
            for itraj, X, Y in reader.iterator(stride=stride, lag=lag):
                collected.append(X)
                collected_lagged.append(Y)
            assert collected
            assert collected_lagged
            assert len(collected) == len(collected_lagged)
            collected = np.vstack(collected)
            collected_lagged = np.vstack(collected_lagged)
            np.testing.assert_allclose(data[::stride][0:len(collected_lagged)], collected, atol=self.eps,
                                                    err_msg="lag={}, stride={}, cs={}".format(lag, stride, chunksize
                                                 ))
            np.testing.assert_allclose(data[lag::stride], collected_lagged, atol=self.eps)
        else:
            collected = []
            for itraj, X in reader.iterator(stride=stride):
                collected.append(X)
            assert collected
            collected = np.vstack(collected)
            np.testing.assert_allclose(data[::stride], collected, atol=self.eps)
            assert itraj == 0 # only one trajectory

    def _test_base_reader(self, file_format, stride, skip, chunksize, transform):
        trajs = self.test_trajs[file_format]
        reader = coor.source(trajs, top=self.pdb_file, chunksize=chunksize)

        if transform == 'identity':
            reader = util.create_transform(reader)

        if chunksize is not None:
            np.testing.assert_equal(reader.chunksize, chunksize)

        it = reader.iterator(stride=stride, skip=skip, lag=0, chunk=chunksize)

        assert it.chunksize is not None
        if chunksize is None:
            max_frames = max_chunksize_from_config(reader.output_type().itemsize)
            assert it.chunksize <= max_frames
            # now we set the chunksize to max_frames, to be able to compare the actual shapes of iterator output.
            chunksize = max_frames

        traj_data = [data[skip::stride] for data in self.traj_data]
        valid_itraj = [i for i, x in enumerate(traj_data) if len(x) > 0]
        output = defaultdict(list)

        with it:
            current_itraj = None
            t = t_total = 0
            for itraj, chunk in it:
                # reset t upon next trajectory
                if itraj != current_itraj:
                    current_itraj = itraj
                    t = 0

                assert len(chunk) <= chunksize or chunksize == 0, '%s' % it
                if chunksize != 0 and len(traj_data[itraj]) - t >= chunksize:
                    assert len(chunk) == chunksize
                elif chunksize == 0:
                    assert len(chunk) == len(traj_data[itraj])

                output[itraj].append(chunk)

                t += len(chunk)
                t_total += len(chunk)

            for itraj in valid_itraj:
                assert itraj in output.keys()

            for itraj in output.keys():
                assert itraj in valid_itraj
                output[itraj] = np.vstack(output[itraj])
                np.testing.assert_allclose(output[itraj], traj_data[itraj], atol=self.eps)

            assert t_total == sum(len(x) for x in output.values())
            assert t_total == reader.n_frames_total(stride=stride, skip=skip)

    def _test_base_reader_with_random_access_stride(self, file_format, stride, chunksize):
        trajs = self.test_trajs[file_format]
        reader = coor.source(trajs, top=self.pdb_file, chunksize=chunksize)

        if chunksize is not None:
            np.testing.assert_equal(reader.chunksize, chunksize)

        it = reader.iterator(stride=stride, lag=0, chunk=chunksize)

        assert it.chunksize is not None
        if chunksize is None:
            max_frames = max_chunksize_from_config(reader.output_type().itemsize)
            assert it.chunksize <= max_frames
            # now we set the chunksize to max_frames, to be able to compare the actual shapes of iterator output.
            chunksize = max_frames
        traj_data = [data[stride[stride[:, 0] == i][:, 1]] for i, data in enumerate(self.traj_data)]

        with it:
            current_itraj = None
            t = t_total = 0
            for itraj, chunk in it:
                # reset t upon next trajectory
                if itraj != current_itraj:
                    current_itraj = itraj
                    t = 0

                assert len(chunk) <= chunksize or chunksize == 0
                if chunksize != 0 and len(traj_data[itraj]) - t >= chunksize:
                    assert len(chunk) == chunksize
                elif chunksize == 0:
                    assert len(chunk) == len(traj_data[itraj])

                np.testing.assert_allclose(chunk, traj_data[itraj][t:t+len(chunk)], atol=self.eps)

                t += len(chunk)
                t_total += len(chunk)
            assert t_total == reader.n_frames_total(stride=stride, skip=0) == it.n_frames_total()


if __name__ == '__main__':
    unittest.main()
