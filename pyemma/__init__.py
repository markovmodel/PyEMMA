
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

from ._base.serialization import load, list_models


def _version_check(current, testing=False):
    """ checks latest version online from http://emma-project.org.

    Can be disabled by setting config.check_version = False.

    >>> from unittest.mock import patch
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
    import os

    from distutils.version import LooseVersion as parse
    from contextlib import closing
    import threading
    import uuid

    import sys
    if 'pytest' in sys.modules or os.getenv('CI', False):
        testing = True

    def _impl():
        import warnings
        from urllib.request import Request, urlopen

        try:
            r = Request('http://emma-project.org/versions.json',
                        headers={'User-Agent': 'PyEMMA-{emma_version}-Py-{python_version}-{platform}-{addr}'
                        .format(emma_version=current, python_version=platform.python_version(),
                                platform=platform.platform(terse=True), addr=uuid.getnode())} if not testing else {})
            with closing(urlopen(r, timeout=30)) as response:
                payload = str(response.read(), encoding='ascii')
            versions = json.loads(payload)
            latest_json = tuple(filter(lambda x: x['latest'], versions))[0]['version']
            latest = parse(latest_json)
            if parse(current) < latest:
                warnings.warn("You are not using the latest release of PyEMMA."
                              " Latest is {latest}, you have {current}."
                              .format(latest=latest, current=current), category=UserWarning)
            if sys.version_info[0] < 3:
                warnings.warn("Python 2.7 usage is deprecated. "
                              "Future versions of PyEMMA will not support it. "
                              "Please upgrade your Python installation.", category=UserWarning)
        except Exception:
            import logging
            logging.getLogger('pyemma').debug("error during version check", exc_info=True)
    return threading.Thread(target=_impl)


# start check in background
_version_check(version).start()
