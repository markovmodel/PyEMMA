#!/usr/bin/env python

try:
    # prefer setuptools in favour of distutils
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
      name='Emma2',
      version='2.0',
      description='EMMA 2',
      url='http://compmolbio.biocomputing-berlin.de/index.php',
      # package_dir={'emma2' : ''},  # all packages are in 'src/emma2' folder
      packages=['emma2', 'emma2.msm'],  # list packages here
      # requires=['Pycluster']  # example dependency to Pycluster package, which may be used for kmeans etc.
      #test_suite='test.test_msmanalyze'
)
