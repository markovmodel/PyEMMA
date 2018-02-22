from __future__ import absolute_import

import unittest
import tempfile
import shutil
import itertools

from coordinates.tests.util import parameterize_test


class TestReaders(unittest.TestCase):

    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp("test-api-src")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    @parameterize_test(itertools.product(
        ("hdf5", "csv", "in-memory", "numpy"), (0, None, 1, 5, 10, 100, 10000), (0, 1, 10, 100, 1000)
    ))
    def test_base_reader(self, name, chunksize, lag):
        pass


if __name__ == '__main__':
    unittest.main()
