
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

'''
Created on 11.06.2015

@author: marscher
'''

from __future__ import absolute_import
import warnings
import unittest
import os
import sys

from pyemma.util.config import readConfiguration
from pyemma.util.files import TemporaryDirectory


class TestConfig(unittest.TestCase):

    @unittest.skipIf(sys.platform == 'win32', 'unix based test')
    def test_can_not_create_cfg_dir(self):
        os.environ['HOME'] = '/dev/null'

        exp_homedir = os.path.expanduser('~')
        assert exp_homedir == '/dev/null'

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            readConfiguration()
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "could not create" in str(w[-1].message)

    @unittest.skipIf(sys.platform == 'win32', 'unix based test')
    def test_non_writeable_cfg_dir(self):

        with TemporaryDirectory() as tmp:
            cfg_dir = os.path.join(tmp, '.pyemma')
            os.mkdir(cfg_dir)
            os.environ['HOME'] = tmp
            # make cfg dir non-writeable
            os.chmod(cfg_dir, 444)

            exp_homedir = os.path.expanduser('~')
            assert exp_homedir == tmp

            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                # Trigger a warning.
                readConfiguration()
                assert len(w) == 1
                assert issubclass(w[-1].category, UserWarning)
                assert "is not writeable" in str(w[-1].message)

    def test_shortcuts(self):
        import pyemma
        pyemma.util.config.show_progress_bars = False
        assert pyemma.config.show_progress_bars == False

    def test_shortcuts2(self):
        import pyemma
        pyemma.config.show_progress_bars = 'True'
        assert pyemma.config.show_progress_bars == 'True'

    def test_shortcut3(self):
        import pyemma
        pyemma.config['show_progress_bars'] = 'True'
        assert pyemma.config.show_progress_bars == 'True'


if __name__ == "__main__":
    unittest.main()