#!/usr/bin/env python

__version__ = 2.0

# prefer setuptools in favour of distutils
try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

def setupPyStallone():
    try:
        from jcc import cpp as cpp
    except ImportError:
        raise RuntimeError('apache jcc not available')
    
    stallone_api_jar = 'lib/stallone/stallone-1.0-SNAPSHOT-api.jar'
    stallone_whole_in_one_jar = \
        'lib/stallone/stallone-1.0-SNAPSHOT-jar-with-dependencies.jar'
    args = ['--jar', stallone_api_jar,
         '--package', 'stallone.mc',
         '--package', 'stallone.algebra',
         '--package', 'stallone.cluster',
         '--include', stallone_whole_in_one_jar,
         #'--use_full_names',
         '--python', 'stallone',
         '--version', '1.0',
         '--reserved', 'extern',
         #'--use-distutils',
         '--egg-info',
         '--files', '2', '--build', '--bdist']
    args.insert(0, __file__)
    cpp.jcc(args)

try:
    import stallone
    print "stallone module found. Not Installing"
    # FIXME: add a parameter to this script to trigger reinstallation of stallone
    if False: # change this to True to force reinstallation
        print "forcing reinstallation of stallone."
        setupPyStallone()
except ImportError:
    setupPyStallone()

setup(
      name = 'Emma2',
      version = __version__,
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The emma2 team',
      # list packages here
      packages = ['emma2',
                  'emma2.msm',
                  'emma2.msm.analysis',
                  'emma2.msm.analysis.dense',
                  'emma2.msm.analysis.sparse',
                  'emma2.msm.estimation',
                  'emma2.msm.estimation.sparse',
                  'emma2.msm.io',
                  'emma2.msm.shared',
                  #'emma2.pmm',
                  'emma2.util'],
      # runtime dependencies
      install_requires = ['numpy >=1.7',
                         'scipy >=0.11',
                         'JCC >=2.17'],
      # build time dependencies
      requires = ['JCC (>=2.17)']
)

