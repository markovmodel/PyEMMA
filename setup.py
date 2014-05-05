#!/usr/bin/env python
"""
EMMA2 setup
"""
import sys

if len(sys.argv) < 2:
    """
        in case we have no args let distutils setup show a basic error message
    """
    from distutils.core import setup
    setup()

import os
import subprocess

"""
we are using setuptools via the bootstrapper ez_setup
"""
from ez_setup import use_setuptools
use_setuptools(version="3.4.4")
from setuptools import setup, Extension, find_packages

"""
################################################################################
    EMMA2 Setup
################################################################################
"""
VERSION = "2.0.0"
ISRELEASED = False

# taken from numpy setup
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'describe'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION

if not ISRELEASED and os.path.exists('.git'):
    __version__ = VERSION + '-' + git_version()
else:
    __version__ = VERSION


cocovar_module = Extension('cocovar', sources = ['extensions/cocovar.c'])

from distutils.command.build_ext import build_ext
class np_build(build_ext):
    """
    Sets numpy include path for extensions. Its ensured, that numpy exists
    at runtime. Note that this workaround seems to disable the ability to
    add additional include dirs via the setup(include_dirs=['...'] option.
    So add them here!
    """
    def initialize_options(self):
        build_ext.initialize_options(self)
        # https://stackoverflow.com/questions/21605927/why-doesnt-setup-requires-work-properly-for-numpy
        __builtins__.__NUMPY_SETUP__ = False
        from numpy import get_include
        self.include_dirs = get_include()

setup(name = 'Emma2',
      version = __version__,
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The Emma2 team',
      # packages are found if their folder contains an __init__.py,
      packages = find_packages(),
      scripts = ['scripts/mm_tica',
                 'scripts/mm_acf',
                 'scripts/mm_project'],
      cmdclass = dict(build_ext = np_build),
      ext_modules = [cocovar_module],
      # FIXME: this goes to egg meta info directory and is not found during init
      data_files = [('emma2', ['emma2.cfg']),
                    # TODO: make this somehow choose the latest version available.
                    ('lib/stallone',
                     ['lib/stallone/stallone-1.0-SNAPSHOT-jar-with-dependencies.jar'])],
      setup_requires = ['numpy >= 1.6.0'],
      # runtime dependencies
      install_requires = ['numpy >= 1.6.0',
                         'scipy >= 0.11',
                         'JPype1 >= 0.5.5'],
)
