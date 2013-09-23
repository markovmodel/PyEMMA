#!/usr/bin/env python

from distutils.core import setup

setup(name='emma2',
      version='2.0',
      description='EMMA 2',
      #url='http://www.python.org/sigs/distutils-sig/',

      package_dir={'' : 'src'}, # all packages are in 'src' folder
      packages=['disc'], # list packages here
      requires=['Pycluster'] # example dependency to Pycluster package, which may be used for kmeans etc.
     )
