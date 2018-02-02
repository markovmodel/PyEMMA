
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import tempfile
import unittest

import six

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


@unittest.skipIf(six.PY2, 'only py3')
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
