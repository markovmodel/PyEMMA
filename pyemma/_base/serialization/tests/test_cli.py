import tempfile
import unittest

from pyemma._base.serialization.cli import main
from pyemma.coordinates import source, tica, cluster_kmeans

import sys
from io import StringIO
from contextlib import contextmanager


@contextmanager
def capture(command, *args, **kwargs):
    out, sys.stdout = sys.stdout, StringIO()
    try:
        command(*args, **kwargs)
        sys.stdout.seek(0)
        yield sys.stdout.read()
    finally:
        sys.stdout = out


class TestListModelCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyemma.datasets import get_bpti_test_data

        d = get_bpti_test_data()
        trajs, top = d['trajs'], d['top']
        s = source(trajs, top=top)

        t = tica(s, lag=1)

        c = cluster_kmeans(t)
        cls.model_file = tempfile.mktemp()
        c.save(cls.model_file, save_streaming_chain=True)

    @classmethod
    def tearDownClass(cls):
        import os
        os.unlink(cls.model_file)

    def test_recursive(self):
        with capture(main, ['--recursive', self.model_file]) as out:
            assert out


if __name__ == '__main__':
    unittest.main()
