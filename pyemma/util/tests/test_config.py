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

from __future__ import absolute_import, print_function

import os
import sys
import unittest

from six.moves import configparser

from pyemma.util.files import TemporaryDirectory
from pyemma.util.exceptions import ConfigDirectoryException
import pkg_resources
import pyemma


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config_inst = pyemma.config()

    def tearDown(self):
        try:
            del os.environ['PYEMMA_CFG_DIR']
        except KeyError:
            pass

    def test_config_vals_match_properties_in_wrapper(self):
        with TemporaryDirectory() as td:
            self.config_inst.cfg_dir = td
            self.assertEqual(self.config_inst.cfg_dir, td)
            from pyemma import config as config_module
            assert hasattr(config_module, 'default_config_file')
            my_cfg = os.path.join(td, 'pyemma.cfg')
            self.assertEqual(pkg_resources.resource_filename('pyemma', 'pyemma.cfg'),
                             config_module.default_config_file)
            reader = configparser.ConfigParser()
            reader.read(my_cfg)

            opts = sorted(reader.options('pyemma'))
            actual = sorted(config_module.keys())
            self.assertEqual(opts, actual)

    @unittest.skipIf(sys.platform == 'win32', 'unix based test')
    def test_can_not_create_cfg_dir(self):
        with self.assertRaises(ConfigDirectoryException) as cm:
            self.config_inst.cfg_dir = '/dev/null'
        self.assertIn("no valid directory", str(cm.exception))

    @unittest.skipIf(sys.platform == 'win32', 'unix based test')
    def test_non_writeable_cfg_dir(self):
        with TemporaryDirectory() as tmp:
            # make cfg dir non-writeable
            os.chmod(tmp, 0x300)
            assert not os.access(tmp, os.W_OK)

            with self.assertRaises(ConfigDirectoryException) as cm:
                self.config_inst.cfg_dir = tmp
            self.assertIn("is not writeable", str(cm.exception))

    def test_shortcuts(self):
        self.config_inst.show_progress_bars = False
        assert not self.config_inst.show_progress_bars

    def test_shortcuts2(self):
        self.config_inst.show_progress_bars = 'True'
        assert self.config_inst.show_progress_bars

    def test_shortcut3(self):
        self.config_inst['show_progress_bars'] = 'True'
        assert self.config_inst.show_progress_bars

    def test_types(self):
        self.config_inst.show_progress_bars = 0
        assert isinstance(self.config_inst.show_progress_bars, bool)

    def test_keys(self):
        self.config_inst.keys()

    def test_set(self):
        self.config_inst.show_progress_bars = 'False'
        self.config_inst.use_traj_length_cache = 0

    def test_save_load_user_cfg_file(self):
        # replace a value with a non default value:
        self.config_inst.show_progress_bars = not self.config_inst.show_progress_bars
        import tempfile
        from six.moves import configparser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.close()
            self.config_inst.save(f.name)
            cfg = configparser.ConfigParser()
            cfg.read(f.name)
            self.assertEqual(cfg.getboolean('pyemma', 'show_progress_bars'), self.config_inst.show_progress_bars)

    def test_save_load_no_cfg_file_given(self):
        """ test that in case no cfg dir has been set, the default location is being used and values changed at
        runtime are used afterwards."""
        # replace a value with a non default value:
        with TemporaryDirectory() as td:
            os.environ['PYEMMA_CFG_DIR'] = td
            self.config_inst = pyemma.config()
            self.config_inst.show_progress_bars = not self.config_inst.show_progress_bars
            self.config_inst.save()

            supposed_to_use_cfg = os.path.join(td, self.config_inst.DEFAULT_CONFIG_FILE_NAME)

            cfg = configparser.RawConfigParser()
            cfg.read(supposed_to_use_cfg)
            self.assertEqual(cfg.getboolean('pyemma', 'show_progress_bars'),
                             self.config_inst.show_progress_bars)

    def test_load(self):
        with TemporaryDirectory() as td:
            new_file = os.path.join(td, "test.cfg")
            self.config_inst.show_progress_bars = not self.config_inst.show_progress_bars
            old_val = self.config_inst.show_progress_bars
            self.config_inst.save(new_file)

            # set a runtime value, differing from what used to be state before save
            self.config_inst.show_progress_bars = not self.config_inst.show_progress_bars

            self.config_inst.load(new_file)
            self.assertEqual(self.config_inst.show_progress_bars, old_val)

    def test_interpolation_from_multiple_files(self):
        # TODO: impl
        pass

    def test_traj_info_max_entries(self):
        assert isinstance(self.config_inst.traj_info_max_entries, int)
        self.config_inst.traj_info_max_entries = 1
        self.assertEqual(self.config_inst.traj_info_max_entries, 1)


if __name__ == "__main__":
    unittest.main()
