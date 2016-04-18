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
Created on 09.03.2016

@author: marscher
'''
from tempfile import NamedTemporaryFile
import sys
import unittest
import logging


from pyemma.util import log
from pyemma.util import config
import six
if six.PY2:
    try:
       import mock
    except ImportError:
       have_mock = False
    else:
       have_mock = True
else:
    from unittest import mock
    have_mock = True

@unittest.skipIf(not have_mock, "dont have mock library")
class TestNonWriteableLogFile(unittest.TestCase):

    def tearDown(self):
        # reset logging
        log.setupLogging(config)

    @unittest.skipIf('win32' in sys.platform, "disabled on win")
    def test(self):
        conf = b"""
# do not disable other loggers by default.
disable_existing_loggers: False

# please do not change version, it is an internal variable used by Python.
version: 1

handlers:
    rotating_files:
        class: logging.handlers.RotatingFileHandler
        filename: /pyemma.log

loggers:
    pyemma:
        level: INFO
        handlers: [rotating_files]
        """
        with NamedTemporaryFile(delete=False) as f:
            f.write(conf)
            f.close()
            with mock.patch('pyemma.util.log.open', create=True) as mock_open:
                mock_open.return_value = open(f.name)

                log.setupLogging(config)
                assert logging.getLogger('pyemma').handlers[0].baseFilename.startswith(config.cfg_dir)

if __name__ == "__main__":
    unittest.main()
