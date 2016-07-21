
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

r"""
=======================================
PyEMMA - Emma's Markov Model Algorithms
=======================================
"""
from __future__ import absolute_import

# set version from versioneer.
from ._version import get_versions
__version__ = get_versions()['version']
version = __version__
del get_versions

from .util import config

from . import coordinates
from . import msm
from . import util
from . import plots
from . import thermo


def _setup_testing():
    # setup function for testing
    from pyemma.util import config
    # do not cache trajectory info in user directory (temp traj files)
    config.use_trajectory_lengths_cache = False
    config.show_progress_bars = False

import unittest as _unittest
# override unittests base class constructor to achieve same behaviour without nose.
_old_init = _unittest.TestCase.__init__
def _new_init(self, *args, **kwargs):
    _old_init(self, *args, **kwargs)
    _setup_testing()

_unittest.TestCase.__init__ = _new_init

