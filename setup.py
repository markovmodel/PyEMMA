#!/usr/bin/env python
"""
EMMA: Emma's Markov Model Algorithms

EMMA is a collection of algorithms implemented mostly in
`NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org>`_
to analyze trajectories generated from any kind of simulation
(e.g. molecular trajectories) via Markov state models (MSM).

TODO: extend.
"""

DOCLINES = __doc__.split("\n")
__requires__ = 'setuptools>=2.2'

import sys
import os
import warnings
import versioneer

from glob import glob

CLASSIFIERS = """\
Development Status :: 4 - Beta
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX :: Linux
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics

"""
from setup_util import getSetuptoolsError
try:
    from setuptools import setup, Extension, find_packages
    from pkg_resources import VersionConflict
except ImportError as ie:
    print getSetuptoolsError()
    sys.exit(23)
# this should catch pkg_resources.DistributionNotFound, which is not
# importable now.
except:
    print "Your version of setuptools is too old. We require at least %s\n" \
          % __requires__
    print getSetuptoolsError()
    sys.exit(24)

versioneer.VCS = 'git'
versioneer.versionfile_source = 'emma2/_version.py'
versioneer.versionfile_build = 'emma2/_version.py'
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'emma2-'  # dirname like 'myproject-1.2.0'

###############################################################################
# Extensions
###############################################################################
def extensions():
    USE_CYTHON = False
    try:
        import Cython
        from Cython.Build import cythonize
        from distutils.version import StrictVersion

        if StrictVersion(Cython.__version__) < StrictVersion('0.20'):
            warnings.warn("Your cython version is too old. Setup will treat this"
                          " as if there is no cython installation and "
                          "use pre-cythonized files.")
        else:
            USE_CYTHON = True
    except ImportError:
        pass

    # this should be obsolete with usage of setuptools; test it
    if USE_CYTHON:
        ext = '.pyx'
    else:
        ext = '.c'

    # setup OpenMP support
    from setup_util import detect_openmp
    openmp_enabled, needs_gomp = detect_openmp()

    # define extensions
    cocovar_module = Extension('emma2.coordinates.transform.cocovar',
                               sources=['emma2/coordinates/transform/cocovar.c'])

    mle_trev_given_pi_dense_module = \
        Extension('emma2.msm.estimation.dense.mle_trev_given_pi',
                  sources=['emma2/msm/estimation/dense/mle_trev_given_pi' + ext,
                           'emma2/msm/estimation/dense/_mle_trev_given_pi.c'],
                  include_dirs=[os.path.abspath('emma2/msm/estimation/dense')],
                  extra_compile_args=['-march=native'])

    mle_trev_given_pi_sparse_module = \
        Extension('emma2.msm.estimation.sparse.mle_trev_given_pi',
                  sources=['emma2/msm/estimation/sparse/mle_trev_given_pi' + ext,
                           'emma2/msm/estimation/sparse/_mle_trev_given_pi.c'],
                  include_dirs=[os.path.abspath('emma2/msm/estimation/dense')],
                  extra_compile_args=['-march=native'])

    mle_trev_sparse_module = \
        Extension('emma2.msm.estimation.sparse.mle_trev',
                  sources=['emma2/msm/estimation/sparse/mle_trev' + ext,
                           'emma2/msm/estimation/sparse/_mle_trev.c'],
                  extra_compile_args=['-march=native'])

    mle_trev_module = [mle_trev_given_pi_dense_module,
                       mle_trev_given_pi_sparse_module,
                       mle_trev_sparse_module]

    if USE_CYTHON:
        mle_trev_module = cythonize(mle_trev_module)

    exts = [cocovar_module] + mle_trev_module

    if openmp_enabled:
        omp_compiler_args = ['-fopenmp']
        omp_libraries = ['-lgomp'] if needs_gomp else []
        omp_defines = [('USE_OPENMP', None)]
        for e in exts:
            e.extra_compile_args += omp_compiler_args
            e.extra_link_args += omp_libraries
            e.define_macros += omp_defines

    return exts

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
        result.append((os.path.join(dest, root), ipynb))

    return result


# installation destination of ipython notebooks for users
# TODO: maybe give user opportunity to specify install location
data_files = []
if os.getenv('INSTALL_IPYTHON', False) or 'install' in sys.argv:
    dest = os.path.join(os.path.expanduser('~'), 'emma2-ipython')
    data_files.extend(ipython_notebooks_mapping(dest))

def get_cmdclass():
    from setuptools.command.build_ext import build_ext
    class np_build(build_ext):
        """
        Sets numpy include path for extensions. Its ensured, that numpy exists
        at runtime. Note that this workaround seems to disable the ability to
        add additional include dirs via the setup(include_dirs=['...'] option.
        So add them here!
        """
        def initialize_options(self):
            # self.include_dirs = [] # gets overwritten by super init
            build_ext.initialize_options(self)
            # https://stackoverflow.com/questions/21605927/why-doesnt-setup-requires-work-properly-for-numpy
            try:
                __builtins__.__NUMPY_SETUP__ = False
            except AttributeError:
                # this may happen, if numpy requirement is already fulfilled.
                pass
            from numpy import get_include

            self.include_dirs = []
            self.include_dirs.append(get_include())

    cmdclass = dict(build_ext=np_build,
                    version=versioneer.cmd_version,
                    versioneer=versioneer.cmd_update_files,
                    build=versioneer.cmd_build,
                    sdist=versioneer.cmd_sdist,
                    )
    return cmdclass

metadata = dict(
    name='Emma2',
    maintainer='Martin K. Scherer',
    maintainer_email='m.scherer@fu-berlin.de',
    author='The Emma2 team',
    # TODO: add this
    author_email='',
    url='http://compmolbio.biocomputing-berlin.de/index.php',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    # packages are found if their folder contains an __init__.py,
    packages=find_packages(),
    # install default emma.cfg into package.
    package_data=dict(emma2=['emma2.cfg']),
    data_files=data_files,
    scripts=[s for s in glob('scripts/*') if s.find('mm_') != -1],
    cmdclass=get_cmdclass(),
    tests_require=['nose'],
    test_suite='nose.collector',
    # runtime dependencies
    install_requires=['numpy>=1.6.0',
                      'scipy>=0.11',
                      'pystallone>=1.0-SNAPSHOT.2'],
)

if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean'))):
    # only require numpy and extensions in case of building/installing
    metadata['ext_modules'] = extensions()
    # setuptools>=2.2 can handle setup_requires
    metadata['setup_requires'] = ['numpy>=1.6.0', 'setuptools>3.6']

try:
    setup(**metadata)
except VersionConflict as ve:
    print ve
    print "You need to manually upgrade your 'setuptools' installation!"
    " Please use these instructions to perform an upgrade and/or consult\n"
    " https://pypi.python.org/pypi/setuptools#installation-instructions"
    print getSetuptoolsError()
