#!/usr/bin/env python

# prefer setuptools in favour of distutils
try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
      name='Emma2',
      version='2.0',
      description='EMMA 2',
      url='http://compmolbio.biocomputing-berlin.de/index.php',
      # think about this... maybe we can reduce redundancy in packages listing
      # package_dir={'emma2' : ''},  # all packages are in 'src/emma2' folder
      
      # list packages here
      packages=['emma2',
                'emma2.msm',
                'emma2.pmm'],
      # runtime dependencies
      requires=['numpy'],
      test_suite='emma2.test.testsuite'
)
