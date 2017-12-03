# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import unittest

from pyemma._base.progress import ProgressReporterMixin, ProgressReporter
from pyemma.util.contexts import settings
import pyemma


class TestProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys, os
        cls.old_std_err = sys.stderr
        sys.stderr = os.devnull

    @classmethod
    def tearDownClass(cls):
        pyemma.config.show_progress_bars = False
        import sys
        sys.stderr = cls.old_std_err

    def setUp(self):
        pyemma.config.show_progress_bars = True
        self.pg = ProgressReporterMixin()
        self.pg._progress_register(100, "test")

    def test_config_override(self):
        self.pg.show_progress = True
        with settings(show_progress_bars=False):
            assert self.pg.show_progress == False

    def test_config_2(self):
        self.pg.show_progress = False
        with settings(show_progress_bars=True):
            assert not self.pg.show_progress

    def test_ctx(self):
        pg = ProgressReporter()
        pg.register(100, 'test')
        pg.register(40, 'test2')
        try:
            with pg.context():
                pg.update(50, stage='test')
                raise Exception()
        except Exception:
            assert pg.num_registered == 0

    def test_ctx2(self):
        pg = ProgressReporter()
        assert pg.show_progress
        pg.register(100, stage='test')
        pg.register(40, stage='test2')
        try:
            with pg.context(stage='test'):
                pg.update(50, stage='test')
                raise Exception()
        except Exception:
            assert pg.num_registered == 1
            assert 'test2' in pg.registered_stages

    def test_ctx3(self):
        pg = ProgressReporter()
        assert pg.show_progress
        pg.register(100, stage='test')
        pg.register(40, stage='test2')
        pg.register(25, stage='test3')
        try:
            with pg.context(stage=('test', 'test3')):
                pg.update(50, stage='test')
                pg.update(2, stage='test3')
                raise Exception()
        except Exception:
            assert pg.num_registered == 1
            assert 'test2' in pg.registered_stages

    def test_ctx4(self):
        pg = ProgressReporter()
        pg.register(100, 'test')
        pg.register(40, 'test2')
        try:
            with pg.context():
                pg.update(50, stage='all')
                raise Exception()
        except Exception:
            assert pg.num_registered == 0

    def test_below_threshold(self):
        # show not raise
        pg = ProgressReporter()
        pg.register(2)
        pg.update(1)
        pg.set_description('dummy')


if __name__ == '__main__':
    unittest.main()
