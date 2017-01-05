
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


def _version_check(current, testing=False):
    """ checks latest version online from http://emma-project.org.

    Can be disabled by setting config.check_version = False.

    >>> from mock import patch
    >>> import warnings, pyemma
    >>> with warnings.catch_warnings(record=True) as cw, patch('pyemma.version', '0.1'):
    ...     warnings.simplefilter('always', UserWarning)
    ...     v = pyemma.version
    ...     t = pyemma._version_check(v, testing=True)
    ...     t.start()
    ...     t.join()
    ...     assert cw, "no warning captured"
    ...     assert "latest release" in str(cw[0].message), "wrong msg"
    """
    if not config.check_version:
        class _dummy:
            def start(self): pass
        return _dummy()
    import json
    import platform
    import six
    import os
    from six.moves.urllib.request import urlopen, Request
    from distutils.version import LooseVersion as parse
    from contextlib import closing
    import threading
    import uuid

    import sys
    if 'pytest' in sys.modules or os.getenv('CI', False):
        testing = True

    def _impl():
        try:
            r = Request('http://emma-project.org/versions.json',
                        headers={'User-Agent': 'PyEMMA-{emma_version}-Py-{python_version}-{platform}-{addr}'
                        .format(emma_version=current, python_version=platform.python_version(),
                                platform=platform.platform(terse=True), addr=uuid.getnode())} if not testing else {})
            encoding_args = {} if six.PY2 else {'encoding': 'ascii'}
            with closing(urlopen(r, timeout=30)) as response:
                payload = str(response.read(), **encoding_args)
            versions = json.loads(payload)
            latest_json = tuple(filter(lambda x: x['latest'], versions))[0]['version']
            latest = parse(latest_json)
            if parse(current) < latest:
                import warnings
                warnings.warn("You are not using the latest release of PyEMMA."
                              " Latest is {latest}, you have {current}."
                              .format(latest=latest, current=current), category=UserWarning)
        except Exception:
            import logging
            logging.getLogger('pyemma').exception("error during version check")
    return threading.Thread(target=_impl)

# start check in background
_version_check(version).start()
