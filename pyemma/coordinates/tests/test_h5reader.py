import unittest
import numpy as np

from pyemma.coordinates.data.h5_reader import H5Reader
from pyemma.util.testing_tools import MockLoggingHandler

try:
    import h5py, tables
    have_hdf5 = True
except ImportError:
    have_hdf5 = False


@unittest.skipIf(not have_hdf5, 'no hdf5 support. Install h5py and pytables')
class TestH5Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        # create test data sets
        cls.directory = tempfile.mkdtemp('test_h5reader')

        cls.f1 = tempfile.mktemp(suffix='.h5', dir=cls.directory)
        cls.shape = (10, 3)
        cls.data = np.arange(cls.shape[0]*cls.shape[1]).reshape(cls.shape)
        with h5py.File(cls.f1, mode='w') as f:
            ds = f.create_group('test').create_group('another_group').create_dataset('test_ds', shape=cls.shape)
            ds[:] = cls.data
            ds = f.create_group('test2').create_dataset('test_ds', shape=cls.shape)
            ds[:] = cls.data
            f.create_dataset('another_ds', shape=(10, 23))
        cls.total_frames = cls.shape[0] * 2

        cls.reader = H5Reader(cls.f1, selection='/.*/test_ds')

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.directory)

    def test_ndim(self):
        assert self.reader.ndim == self.shape[1]
        assert self.reader.filenames == [self.f1]
        self.assertEqual(self.reader.n_frames_total(), self.total_frames)
        self.assertEqual(self.reader.n_datasets, 2)

    def test_udpate_selection(self):
        self.reader.selection = '/test2/*'
        assert self.reader.ndim == self.shape[1]
        assert self.reader.n_frames_total() == self.shape[0]
        assert self.reader.n_datasets == 1

    def test_non_matching_selection(self):
        with self.assertRaises(ValueError):
            h = MockLoggingHandler()
            import logging
            logging.getLogger('pyemma.coordinates').addHandler(h)
            r = H5Reader(self.f1, selection='/non_existent')

            self.assertIn('did not match', h.messages['warning'])
            assert r.ndim == -1
            assert r.n_frames_total() == 0
            assert r.ntraj == 0

    def test_output(self):
        out = self.reader.get_output()
        np.testing.assert_equal(out, [self.data]*2)

    def test_source(self):
        from pyemma.coordinates import source
        reader = source(self.f1, selection='/.*/test_ds')
        self.assertIsInstance(reader, H5Reader)
        self.assertEqual(reader.ntraj, 2)
