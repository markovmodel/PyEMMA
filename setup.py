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

"""
we are using setuptools via the bootstrapper ez_setup
"""
from ez_setup import use_setuptools
use_setuptools(version="2.1")
from setuptools import __version__ as st_version
print "Using setuptools version: ", st_version
from setuptools import setup, Extension, find_packages

if '--help' in sys.argv:
    sys.exit(0)
"""
################################################################################
    EMMA2 Setup
################################################################################
"""
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
        from numpy import get_include
        self.include_dirs = get_include()

setup(name = 'Emma2',
      version = '2.0',
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
      # TODO: this is a open issue in setuptools: https://bitbucket.org/pypa/setuptools/issue/141/setup_requires-feature-does-not-handle
      #      setup_requires = ['numpy >= 1.8'],
      # runtime dependencies
      install_requires = ['numpy >= 1.6.0',
                         'scipy >= 0.9',
                         'JPype1 >= 0.5.4.5'],
)
