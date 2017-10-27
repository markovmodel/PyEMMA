
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


class TestProgress(unittest.TestCase):

    def setUp(self):
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
        try:
            with pg.context():
                pg.update(50)
                raise Exception()
        except Exception:
            assert len(pg._prog_rep_progressbars) == 0


if __name__ == '__main__':
    unittest.main()
