#!/usr/bin/env python
"""
################################################################################
    EMMA2 Setup
################################################################################
"""
import sys
import os
import subprocess
from glob import glob


# define minimum requirements for our setup script.
__requires__ = 'setuptools >= 3.0.0'

def getSetuptoolsError():
    bootstrap_setuptools = 'python2.7 -c \"import urllib2;\n\
url=\'https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py\';\n\
exec urllib2.urlopen(url).read()\"'
    
    cmd = ((80*'=') + '\n' + bootstrap_setuptools + '\n' +(80*'='))
    s = 'You can use the following command to upgrade/install it:\n%s' % cmd
    return s

try:
    from setuptools import setup, Extension, find_packages
except ImportError as ie:
    print "Sorry, we require %s\n" % __requires__
    print getSetuptoolsError()
    sys.exit(23)
except: # this should catch pkg_resources.DistributionNotFound, which is not importable now.
    print "Your version of setuptools is too old. We require at least %s\n" \
          % __requires__
    print getSetuptoolsError()
    sys.exit(24)


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
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION

if not ISRELEASED and os.path.exists('.git'):
    __version__ = VERSION + '-' + git_version()
else:
    __version__ = VERSION


cocovar_module = Extension('emma2.coordinates.cocovar',
                            sources = ['extensions/cocovar.c'])

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


from setuptools.command.test import test
class DiscoverTest(test):
    def discover_and_run_tests(self):
        import unittest
        # get setup.py directory
        setup_file = sys.modules['__main__'].__file__
        setup_dir = os.path.abspath(os.path.dirname(setup_file))
        # use the default shared TestLoader instance
        test_loader = unittest.defaultTestLoader
        # use the basic test runner that outputs to sys.stderr
        test_runner = unittest.TextTestRunner(verbosity=2)
        # automatically discover all tests
        search_path = os.path.join(setup_dir, 'emma2')
        test_suite = test_loader.discover(search_path, pattern='*_test.py')
        # run the test suite
        test_runner.run(test_suite)

    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # need to reset fake jdk for runtime.
        if fake_jdk:
            os.environ.pop('JAVA_HOME', None)
        self.discover_and_run_tests()

# HACK for JPype installation:
# we do not want the user to have JDK, so we provide jni.h here.
if not os.environ.get('JAVA_HOME', None):
    fake_jdk = True
    os.environ['JAVA_HOME'] = os.path.abspath('lib/stallone/')

metadata = dict(
      name = 'Emma2',
      version = __version__,
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The Emma2 team',
      # packages are found if their folder contains an __init__.py,
      packages = find_packages(),
      # install default emma.cfg and stallone jar into package.
      package_data = {'emma2' : ['emma2.cfg','stallone-1.0-SNAPSHOT-jar-with-dependencies.jar']},
      scripts = [s for s in glob('scripts/*') if s.find('mm_') != -1],
      cmdclass = dict(build_ext = np_build,
                      test = DiscoverTest),
      ext_modules = [cocovar_module],
      setup_requires = ['numpy >= 1.6.0'],
      tests_require = [],
      # runtime dependencies
      install_requires = ['numpy >= 1.6.0',
                         'scipy >= 0.11',
                         'JPype1 >= 0.5.5'],
)

setup(**metadata)