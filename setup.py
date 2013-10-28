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
      author='',
      
      # list packages here
      packages=['emma2',
                'emma2.msm.analysis',
                'emma2.msm.analysis.dense',
                'emma2.msm.analysis.sparse',
                'emma2.msm.estimation',
                'emma2.msm.estimation.sparse',
                'emma2.msm.io',
                'emma2.pmm'],
      # runtime dependencies
      requires=['numpy (>=1.7)', 'scipy (>=0.11)']
)
