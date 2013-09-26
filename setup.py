#!/usr/bin/env python

# prefer setuptools in favour of distutils
try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    """
    little wrapper for pytest for direct usage with setup() function
    """
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
      name='Emma2',
      version='2.0',
      description='EMMA 2',
      url='http://compmolbio.biocomputing-berlin.de/index.php',
      author='',
      
      # list packages here
      packages=['emma2',
                'emma2.msm',
                'emma2.pmm'],
      # runtime dependencies
      requires=['numpy (>=1.7)'],
      # testing dependencies
      tests_require=['pytest'],
      cmdclass = {'test': PyTest},
)
