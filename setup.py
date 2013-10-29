#!/usr/bin/env python

__version__ = 2.0

# prefer setuptools in favour of distutils
try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup
    from distutils.command.install import install as build

# TODO: integrate jcc call in build phase
# create egg and integrate this somehow in build process

def create_packages(packages):
    cp_string = ''.join("--package %s " \
                        % ''.join(map(str, x)) for x in packages)
    return cp_string

class build_JCC_Wrapper(build):
    """
    invokes apache jcc to build a wrapper for the public api of stallone 
    """
    def run(self):
        import sys, shlex, subprocess
        additional_packages = ['stallone.mc', 
                               'stallone.algebra',
                               'stallone.cluster']
        
        stallone_api_jar = 'lib/stallone/stallone-1.0-SNAPSHOT-api.jar'

        call = sys.executable + ' -m jcc --jar ' + stallone_api_jar \
             + ' ' + create_packages(additional_packages) \
             + '--include stallone-1.0-SNAPSHOT-jar-with-dependencies.jar' \
             + ' --use_full_names' \
             + " --python " + __name__ + ' ' \
             + " --version " + str(__version__) + " --reserved extern" \
             + " --module util/ArrayWrapper.py" \
             + ' --files 2 --use_full_names'
        
        print call
        return subprocess.call(shlex.split(call))

setup(
      name = 'Emma2',
      version = __version__,
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The emma2 team',
      cmdclass = dict(install = build_JCC_Wrapper),
      # list packages here
      packages = ['emma2',
                 'emma2.msm.analysis',
                 'emma2.msm.analysis.dense',
                 'emma2.msm.analysis.sparse',
                 'emma2.msm.estimation',
                 'emma2.msm.estimation.sparse',
                 'emma2.msm.io',
                 'emma2.pmm'],
      # runtime dependencies
      install_requires = ['numpy >=1.7',
                         'scipy >=0.11',
                         'JCC >=2.17'],
      # build time dependencies
      requires = ['JCC (>=2.17)']
)
