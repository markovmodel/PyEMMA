#!/usr/bin/env python

__version__ = 2.0

# prefer setuptools in favour of distutils
try:
    from setuptools.core import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from os import environ
from sys import argv
import numpy

def setupPyStallone():
    try:
        from jcc import cpp as cpp
    except ImportError:
        raise RuntimeError('Apache JCC not available! '\
                           'Install with easy_install [--user] JCC')
    
    stallone_api_jar = 'lib/stallone/stallone-1.0-SNAPSHOT-api.jar'
    stallone_whole_in_one_jar = \
        'lib/stallone/stallone-1.0-SNAPSHOT-jar-with-dependencies.jar'
    args = ['--jar', stallone_api_jar,
         '--package', 'stallone.mc',
         '--package', 'stallone.algebra',
         '--package', 'stallone.cluster',
         '--package', 'java.lang',
         '--package', 'java.util',
         '--include', stallone_whole_in_one_jar,
         #'--use_full_names', # does not work...
         '--python', 'stallone', # python module name
         '--version', '1.0',
         '--reserved', 'extern',
         #'--use-distutils',
         #'--egg-info',
         '--output', 'target', # output directory, name 'build' is buggy in
                               # case of setup.py sdist, which does not include stuff from this dirs
         '--files', '2']

    # program name first. (this is needed, as source files of jcc are looked up 
    # relative to this path)
    args.insert(0, cpp.__file__)
    
    if 'sdist' in argv:
        # call the setup once to generate the wrapper code
        cpp.jcc(args)
        
        # now try to build sdist....
        args.append('--egg-info')
        # FIXME: include jars in lib dir as resources for source distribution
        # .....
        #args.append('--resources')
        #args.append('lib/stallone')
        #args.append(stallone_api_jar)
        #args.append('--resources')
        #args.append(stallone_whole_in_one_jar)

        
        # create source dist
        args.append('--extra-setup-arg')
        args.append('sdist')
    else: # we want to build this now.
        args.append('--build')
        args.append('--bdist')
    
    cpp.jcc(args)

try:
    import stallone
    print "stallone module found."
    try:
        environ['REBUILD_STALLONE']
        rebuild = True
    except KeyError:
        rebuild = False
    
    if rebuild:
        print "forcing rebuild of stallone."
        setupPyStallone()
    else:
        print "skipping installation of stallone."
except ImportError:
    setupPyStallone()

cocovar_module = Extension('cocovar', sources = ['extensions/cocovar.c'])

setup(
      name = 'Emma2',
      version = __version__,
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The Emma2 team',
      # list packages here
      packages = ['emma2',
                  'emma2.coordinates',
                  'emma2.msm',
                  'emma2.msm.analysis',
                  'emma2.msm.analysis.dense',
                  'emma2.msm.analysis.sparse',
                  'emma2.msm.estimation',
                  'emma2.msm.estimation.sparse',
                  'emma2.msm.io',
                  'emma2.pmm',
                  'emma2.util'],
      scripts = ['scripts/ImpliedTimescalePlot.py',
                 'scripts/mm_tica',
                 'scripts/mm_acf',
                 'scripts/mm_project'],
      include_dirs = [numpy.get_include()],
      ext_modules = [cocovar_module],
      # FIXME: this goes to egg meta info directory and is not found during init
      data_files = [('emma2', ['emma2.cfg'])],
      # runtime dependencies
      install_requires = ['numpy >=1.7',
                         'scipy >=0.11',
                         'JCC >=2.17'],
      # build time dependencies
      requires = ['JCC (>=2.17)'],
)
