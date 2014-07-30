#!/usr/bin/env python
"""
################################################################################
    EMMA2 Setup
################################################################################
"""
import sys
import os
from glob import glob

# try:
#     from Cython.Build import cythonize
#     USE_CYTHON=True
# except ImportError:
#     USE_CYTHON=False

from Cython.Build import cythonize
USE_CYTHON=True

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

import versioneer
versioneer.versionfile_source = 'emma2/_version.py'
versioneer.versionfile_build = 'emma2/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'emma2-' # dirname like 'myproject-1.2.0'


cocovar_module = Extension('emma2.coordinates.transform.cocovar',
                            sources = ['emma2/coordinates/transform/cocovar.c'])

# see also http://docs.cython.org/src/reference/compilation.html#distributing-cython-modules
# mle_trev_given_pi_dense_module = Extension('emma2.msm.estimation.dense.mle_trev_given_pi', 
#                                            sources=['emma2/msm/estimation/dense/mle_trev_given_pi.pyx', 'emma2/msm/estimation/dense/_mle_trev_given_pi.c'],
#                                            extra_compile_args = ['-fopenmp','-march=native'],
#                                            libraries = ['gomp'])

if USE_CYTHON:
    ext='.pyx'
else:
    ext='.c'

# mle_trev_given_pi_dense_module = Extension('emma2.msm.estimation.dense.mle_trev_given_pi', 
#                                            sources=['emma2/msm/estimation/dense/mle_trev_given_pi'+ext, 
#                                                     'emma2/msm/estimation/dense/_mle_trev_given_pi.c', 
#                                                     'emma2/msm/estimation/dense/_mle_trev_given_pi.h'],
#                                            extra_compile_args = ['-march=native'])

# mle_trev_given_pi_sparse_module = Extension('emma2.msm.estimation.sparse.mle_trev_given_pi', 
#                                             sources=['emma2/msm/estimation/sparse/mle_trev_given_pi'+ext,
#                                                      'emma2/msm/estimation/sparse/_mle_trev_given_pi.c',
#                                                      'emma2/msm/estimation/sparse/_mle_trev_given_pi.h'],
#                                             extra_compile_args = ['-march=native'])

mle_trev_given_pi_dense_module = Extension('emma2.msm.estimation.dense.mle_trev_given_pi', 
                                           sources=['emma2/msm/estimation/dense/mle_trev_given_pi'+ext],
                                           include_dirs=['emma2/msm/estmiation/dense/'],
                                           extra_compile_args = ['-march=native'])

mle_trev_given_pi_sparse_module = Extension('emma2.msm.estimation.sparse.mle_trev_given_pi', 
                                            sources=['emma2/msm/estimation/sparse/mle_trev_given_pi'+ext],
                                            include_dirs=['emma2/msm/estmiation/sparse/'],
                                            extra_compile_args = ['-march=native'])

if USE_CYTHON:
    mle_trev_given_pi_module=cythonize([mle_trev_given_pi_dense_module, mle_trev_given_pi_sparse_module])
else:
    mle_trev_given_pi_module=[mle_trev_given_pi_dense_module, mle_trev_given_pi_sparse_module]

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
        try:
            __builtins__.__NUMPY_SETUP__ = False
        except AttributeError:
            # this may happen, if numpy requirement is already fulfilled.
            pass
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

data_files = []

# installation destination of ipython notebooks for users
# TODO: maybe give user opportunity to specify install location
if os.getenv('INSTALL_IPYTHON', False) or 'install' in sys.argv:
    dest = os.path.join(os.path.expanduser('~'), 'emma2-ipython' ) 
    def ipython_notebooks_mapping(dest):
        """
        returns a mapping for each file in ipython directory:
        [ (dest/$dir, [$file]) ] for each $dir and $file in this dir.
        """
        result = []
        for root, dirs, files in os.walk('ipython'):
            ipynb = []
            for f in files:
                ipynb.append(os.path.join(root, f))
            result.append( (os.path.join(dest, root), ipynb) )
                
        return result
    
    m = ipython_notebooks_mapping(dest)
    data_files.extend(m)

metadata = dict(
      name = 'Emma2',
      version = versioneer.get_version(),
      description = 'EMMA 2',
      url = 'http://compmolbio.biocomputing-berlin.de/index.php',
      author = 'The Emma2 team',
      author_email = '', # TODO: add this
      # packages are found if their folder contains an __init__.py,
      packages = find_packages(),
      # install default emma.cfg and stallone jar into package.
      package_data = {'emma2' : ['emma2.cfg','stallone-1.0-SNAPSHOT-jar-with-dependencies.jar']},
      data_files = data_files,
      scripts = [s for s in glob('scripts/*') if s.find('mm_') != -1],
      cmdclass = dict(build_ext = np_build,
                      test = DiscoverTest,
                      version = versioneer.cmd_version,
                      versioneer = versioneer.cmd_update_files,
                      build = versioneer.cmd_build,
                      sdist = versioneer.cmd_sdist,
                      ),
      #ext_modules = cythonize([cocovar_module,mle_trev_given_pi_module]),
      #ext_modules = [cocovar_module]+cythonize([mle_trev_given_pi_dense_module,mle_trev_given_pi_sparse_module]),
      ext_modules=[cocovar_module]+mle_trev_given_pi_module,
      setup_requires = ['numpy >= 1.6.0'],
      tests_require = [],
      # runtime dependencies
      install_requires = ['numpy >= 1.6.0',
                          'scipy >= 0.11',
                          'JPype1 >= 0.5.5',
                          'cython >=0.20.2'],
)

setup(**metadata)
