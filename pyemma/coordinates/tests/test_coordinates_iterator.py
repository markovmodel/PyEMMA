import unittest
import numpy as np

from pyemma.coordinates.data import DataInMemory
from pyemma.util.files import TemporaryDirectory
import os
from glob import glob


class TestCoordinatesIterator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = [np.random.random((100, 3)) for _ in range(3)]

    def test_current_trajindex(self):
        r = DataInMemory(self.d)
        expected_itraj = 0
        for itraj, X in r.iterator(chunk=0):
            assert itraj == expected_itraj
            expected_itraj += 1

        expected_itraj = -1
        it = r.iterator(chunk=16)
        for itraj, X in it:
            if it.pos == 0:
                expected_itraj += 1
            assert itraj == expected_itraj == it.current_trajindex

    def test_n_chunks(self):
        r = DataInMemory(self.d)

        it0 = r.iterator(chunk=0)
        assert it0._n_chunks == 3  # 3 trajs

        it1 = r.iterator(chunk=50)
        assert it1._n_chunks == 3 * 2  # 2 chunks per trajectory

        it2 = r.iterator(chunk=30)
        # 3 full chunks and 1 small chunk per trajectory
        assert it2._n_chunks == 3 * 4

        it3 = r.iterator(chunk=30)
        it3.skip = 10
        assert it3._n_chunks == 3 * 3  # 3 full chunks per traj

        it4 = r.iterator(chunk=30)
        it4.skip = 5
        # 3 full chunks and 1 chunk of 5 frames per trajectory
        assert it4._n_chunks == 3 * 4

        # test for lagged iterator
        for stride in range(1, 5):
            for lag in range(0, 18):
                it = r.iterator(
                    lag=lag, chunk=30, stride=stride, return_trajindex=False)
                chunks = 0
                for _ in it:
                    chunks += 1
                assert chunks == it._n_chunks

    def test_skip(self):
        r = DataInMemory(self.d)
        lagged_it = r.iterator(lag=5)
        assert lagged_it._it.skip == 0
        assert lagged_it._it_lagged.skip == 5

        it = r.iterator()
        for itraj, X in it:
            if itraj == 0:
                it.skip = 5
            if itraj == 1:
                assert it.skip == 5

    def test_chunksize(self):
        r = DataInMemory(self.d)
        cs = np.arange(1, 17)
        i = 0
        it = r.iterator(chunk=cs[i])
        for itraj, X in it:
            if not it.last_chunk_in_traj:
                assert len(X) == it.chunksize
            else:
                assert len(X) <= it.chunksize
            i += 1
            i %= len(cs)
            it.chunksize = cs[i]
            assert it.chunksize == cs[i]

    def test_last_chunk(self):
        r = DataInMemory(self.d)
        it = r.iterator(chunk=0)
        for itraj, X in it:
            assert it.last_chunk_in_traj
            if itraj == 2:
                assert it.last_chunk

    def test_stride(self):
        r = DataInMemory(self.d)
        stride = np.arange(1, 17)
        i = 0
        it = r.iterator(stride=stride[i], chunk=1)
        for _ in it:
            i += 1
            i %= len(stride)
            it.stride = stride[i]
            assert it.stride == stride[i]

    def test_return_trajindex(self):
        r = DataInMemory(self.d)
        it = r.iterator(chunk=0)
        it.return_traj_index = True
        assert it.return_traj_index is True
        for tup in it:
            assert len(tup) == 2
        it.reset()
        it.return_traj_index = False
        assert it.return_traj_index is False
        itraj = 0
        for tup in it:
            np.testing.assert_equal(tup, self.d[itraj])
            itraj += 1

        for tup in r.iterator(return_trajindex=True):
            assert len(tup) == 2
        itraj = 0
        for tup in r.iterator(return_trajindex=False):
            np.testing.assert_equal(tup, self.d[itraj])
            itraj += 1

    def test_pos(self):
        r = DataInMemory(self.d)
        r.chunksize = 17
        it = r.iterator()
        t = 0
        for itraj, X in it:
            assert t == it.pos
            t += len(X)
            if it.last_chunk_in_traj:
                t = 0

    def test_write_to_csv_propagate_filenames(self):
        from pyemma.coordinates import source, tica
        with TemporaryDirectory() as td:
            data = [np.random.random((20, 3))] * 3
            fns = [os.path.join(td, f)
                   for f in ('blah.npy', 'blub.npy', 'foo.npy')]
            for x, fn in zip(data, fns):
                np.save(fn, x)
            reader = source(fns)
            assert reader.filenames == fns
            tica_obj = tica(reader, lag=1, dim=2)
            tica_obj.write_to_csv(extension=".exotic", chunksize=3)
            res = sorted([os.path.abspath(x) for x in glob(td + os.path.sep + '*.exotic')])
            self.assertEqual(len(res), len(fns))
            desired_fns = sorted([s.replace('.npy', '.exotic') for s in fns])
            self.assertEqual(res, desired_fns)

            # compare written results
            expected = tica_obj.get_output()
            actual = source(list(s.replace('.npy', '.exotic') for s in fns)).get_output()
            assert len(actual) == len(fns)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e)

if __name__ == '__main__':
    unittest.main()
