import unittest
from abc import abstractmethod

import numpy as np

from pyemma.coordinates.data import DataInMemory
from pyemma.coordinates.tests.util import create_top, create_traj_given_xyz
from pyemma.util.files import TemporaryDirectory
import os
from glob import glob


class TestCoordinatesIteratorBase(unittest.TestCase):
    rtol = 1e-7
    atol = 0

    def __init__(self, *args, **kwargs):
        super(TestCoordinatesIteratorBase, self).__init__(*args, **kwargs)
        self.helper = None
        # Kludge alert: We want this class to carry test cases without being run
        # by the unit test framework, so the `run' method is overridden to do
        # nothing.  But in order for sub-classes to be able to do something when
        # run is invoked, the constructor will rebind `run' from TestCase.
        if self.__class__ != TestCoordinatesIteratorBase:
            # Rebind `run' from the parent class.
            self.run = unittest.TestCase.run.__get__(self, self.__class__)
        else:
            self.run = lambda self, *args, **kwargs: None

    @classmethod
    def setUpClass(cls):
        cls.d = [np.random.random((100, 3)) for _ in range(3)]

    def test_current_trajindex(self):

        expected_itraj = 0
        for itraj, X in self.reader.iterator(chunk=0):
            self.assertEqual(itraj, expected_itraj)
            expected_itraj += 1

        expected_itraj = -1
        it = self.reader.iterator(chunk=16)
        for itraj, X in it:
            if it.pos == 0:
                expected_itraj += 1
            self.assertEqual(itraj, expected_itraj)  # ,  it.current_trajindex)

    def test_n_chunks(self):

        it0 = self.reader.iterator(chunk=0)
        self.assertEqual(it0._n_chunks, 3)  # 3 trajs

        it1 = self.reader.iterator(chunk=50)
        self.assertEqual(it1._n_chunks, 3 * 2)  # 2 chunks per trajectory

        it2 = self.reader.iterator(chunk=30)
        # 3 full chunks and 1 small chunk per trajectory
        self.assertEqual(it2._n_chunks, 3 * 4)

        it3 = self.reader.iterator(chunk=30)
        it3.skip = 10
        self.assertEqual(it3._n_chunks, 3 * 3)  # 3 full chunks per traj

        it4 = self.reader.iterator(chunk=30)
        it4.skip = 5
        # 3 full chunks and 1 chunk of 5 frames per trajectory
        self.assertEqual(it4._n_chunks, 3 * 4)

        # test for lagged iterator
        for stride in range(1, 5):
            for lag in range(0, 18):
                it = self.reader.iterator(
                    lag=lag, chunk=30, stride=stride, return_trajindex=False)
                chunks = 0
                for _ in it:
                    chunks += 1
                self.assertEqual(chunks, it._n_chunks, "stride={s}, lag={t}".format(s=stride, t=lag))

    def test_skip(self):
        lagged_it = self.reader.iterator(lag=5)
        self.assertEqual(lagged_it._it.skip, 0)
        self.assertEqual(lagged_it._it_lagged.skip, 5)

        it = self.reader.iterator()
        for itraj, X in it:
            if itraj == 0:
                it.skip = 5
            if itraj == 1:
                self.assertEqual(it.skip, 5)

    def test_skip_with_offset(self):
        skip = 3
        r2 = DataInMemory(self.d)

        desired = r2.get_output(skip=skip)

        self.reader.skip = skip
        it = self.reader.iterator(return_trajindex=True)
        from collections import defaultdict
        out = defaultdict(list)
        for itraj, X in it:
            out[itraj].append(X)

        out = [np.concatenate(chunks) for chunks in out.values()]
        np.testing.assert_allclose(out, desired)

    def test_chunksize(self):
        cs = np.arange(1, 17)
        i = 0
        it = self.reader.iterator(chunk=cs[i])
        for itraj, X in it:
            if not it.last_chunk_in_traj:
                self.assertEqual(len(X), it.chunksize)
            else:
                assert len(X) <= it.chunksize
            i += 1
            i %= len(cs)
            it.chunksize = cs[i]
            self.assertEqual(it.chunksize, cs[i])

    def test_last_chunk(self):
        it = self.reader.iterator(chunk=0)
        for itraj, X in it:
            assert it.last_chunk_in_traj
            if itraj == 2:
                assert it.last_chunk

    def test_stride(self):
        stride = np.arange(1, 17)
        i = 0
        it = self.reader.iterator(stride=stride[i], chunk=1)
        for _ in it:
            i += 1
            i %= len(stride)
            it.stride = stride[i]
            self.assertEqual(it.stride, stride[i])

    def test_return_trajindex(self):
        it = self.reader.iterator(chunk=0)
        it.return_traj_index = True
        assert it.return_traj_index is True
        for tup in it:
            self.assertEqual(len(tup), 2)
        it.reset()
        it.return_traj_index = False
        assert it.return_traj_index is False
        itraj = 0
        for tup in it:
            np.testing.assert_allclose(tup, self.d[itraj])
            itraj += 1

        for tup in self.reader.iterator(return_trajindex=True):
            self.assertEqual(len(tup), 2)
        itraj = 0
        for tup in self.reader.iterator(return_trajindex=False):
            np.testing.assert_allclose(tup, self.d[itraj])
            itraj += 1

    def test_pos(self):
        self.reader.chunksize = 17
        it = self.reader.iterator()
        t = 0
        for itraj, X in it:
            self.assertEqual(t, it.pos)
            t += len(X)
            if it.last_chunk_in_traj:
                t = 0


class TestDataInMem(TestCoordinatesIteratorBase, unittest.TestCase):
    def setUp(self):
        self.reader = DataInMemory(self.d)

    def test_write_to_csv_propagate_filenames(self):
        from pyemma.coordinates import source, tica
        with TemporaryDirectory() as td:
            data = [np.random.random((20, 3))] * 3
            fns = [os.path.join(td, f)
                   for f in ('blah.npy', 'blub.npy', 'foo.npy')]
            for x, fn in zip(data, fns):
                np.save(fn, x)
            self.reader = source(fns)
            self.assertEqual(self.reader.filenames, fns)
            tica_obj = tica(self.reader, lag=1, dim=2)
            tica_obj.write_to_csv(extension=".exotic", chunksize=3)
            res = sorted([os.path.abspath(x) for x in glob(td + os.path.sep + '*.exotic')])
            self.assertEqual(len(res), len(fns))
            desired_fns = sorted([s.replace('.npy', '.exotic') for s in fns])
            self.assertEqual(res, desired_fns)

            # compare written results
            expected = tica_obj.get_output()
            actual = source(list(s.replace('.npy', '.exotic') for s in fns)).get_output()
            self.assertEqual(len(actual), len(fns))
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e)


class TestTrajectoryFormatAbstract(object):
    _format = None

    @classmethod
    def setUpClass(cls):
        cls.d = [np.random.random((100, 3)) for _ in range(3)]

        super(TestTrajectoryFormatAbstract, cls).setUpClass()
        import tempfile
        cls.tdir = tempfile.mkdtemp("test_coor_iter_{}".format(cls._format))
        cls.top = create_top(1)
        cls._trajs = []

    @classmethod
    def tearDownClass(cls):
        import shutil
        print("rm")
        shutil.rmtree(cls.tdir)

    @property
    def trajs(self):
        if self._format is not None:
            self._trajs = [create_traj_given_xyz(xyz=xyz, top=self.top,
                                                 format=self._format, directory=self.tdir) for
                           xyz in self.d]
        return self._trajs

    def setUp(self):
        from pyemma.coordinates.data.feature_reader import FeatureReader
        self.reader = FeatureReader(self.trajs, self.top)


class TestXTC(TestTrajectoryFormatAbstract, TestCoordinatesIteratorBase, unittest.TestCase):
    _format = '.xtc'


class TestDCD(TestTrajectoryFormatAbstract, TestCoordinatesIteratorBase, unittest.TestCase):
    _format = '.dcd'


class TestH5(TestTrajectoryFormatAbstract, TestCoordinatesIteratorBase, unittest.TestCase):
    _format = '.h5'


class TestBinPos(TestTrajectoryFormatAbstract, TestCoordinatesIteratorBase, unittest.TestCase):
    _format = '.binpos'


if __name__ == '__main__':
    unittest.main()
