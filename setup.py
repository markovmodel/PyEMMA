#!/usr/bin/env python

# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""EMMA: Emma's Markov Model Algorithms

EMMA is an open source collection of algorithms implemented mostly in
`NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org>`_
to analyze trajectories generated from any kind of simulation
(e.g. molecular trajectories) via Markov state models (MSM).

"""
# TODO: extend docstring
DOCLINES = __doc__.split("\n")
__requires__ = 'setuptools>=3.6'

import sys
import os
import versioneer
import warnings

CLASSIFIERS = """\
Development Status :: 4 - Beta
Environment :: Console
Environment :: MacOS X
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics

"""
from setup_util import getSetuptoolsError, lazy_cythonize
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

###############################################################################
# Extensions
###############################################################################
def extensions():
    """How do we handle cython:
    1. when on git, require cython during setup time (do not distribute
    generated .c files via git)
     a) cython present -> fine
     b) no cython present -> install it on the fly. Extensions have to have .pyx suffix
    This is solved via a lazy evaluation of the extension list. This is needed,
    because build_ext is being called before cython will be available.
    https://bitbucket.org/pypa/setuptools/issue/288/cannot-specify-cython-under-setup_requires

    2. src dist install (have pre-converted c files and pyx files)
     a) cython present -> fine
     b) no cython -> use .c files
    """
    USE_CYTHON = False
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        warnings.warn('Cython not found. Using pre cythonized files.')

    # setup OpenMP support
    from setup_util import detect_openmp
    openmp_enabled, needs_gomp = detect_openmp()

    import mdtraj

    exts = []

    mle_trev_given_pi_dense_module = \
        Extension('pyemma.msm.estimation.dense.mle_trev_given_pi',
                  sources=['pyemma/msm/estimation/dense/mle_trev_given_pi.pyx',
                           'pyemma/msm/estimation/dense/_mle_trev_given_pi.c'],
                  include_dirs=['pyemma/msm/estimation/dense'])

    mle_trev_given_pi_sparse_module = \
        Extension('pyemma.msm.estimation.sparse.mle_trev_given_pi',
                  sources=['pyemma/msm/estimation/sparse/mle_trev_given_pi.pyx',
                           'pyemma/msm/estimation/sparse/_mle_trev_given_pi.c'],
                  include_dirs=['pyemma/msm/estimation/dense'])

    mle_trev_sparse_module = \
        Extension('pyemma.msm.estimation.sparse.mle_trev',
                  sources=['pyemma/msm/estimation/sparse/mle_trev.pyx',
                           'pyemma/msm/estimation/sparse/_mle_trev.c'])
    if sys.platform.startswith('win'):
        lib_prefix = 'lib'
    else:
        lib_prefix = ''
    regspatial_module = \
        Extension('pyemma.coordinates.clustering.regspatial',
                  sources=[
                      'pyemma/coordinates/clustering/src/regspatial.c',
                      'pyemma/coordinates/clustering/src/clustering.c'],
                  include_dirs=[
                      mdtraj.capi()['include_dir'],
                      'pyemma/coordinates/clustering/include'
                  ],
                  libraries=[lib_prefix+'theobald'],
                  library_dirs=[mdtraj.capi()['lib_dir']],
                  extra_compile_args=['-std=c99'])
    kmeans_module = \
        Extension('pyemma.coordinates.clustering.kmeans_clustering',
                  sources=[
                      'pyemma/coordinates/clustering/src/kmeans.c',
                      'pyemma/coordinates/clustering/src/clustering.c'],
                  include_dirs=[
                      mdtraj.capi()['include_dir'],
                      'pyemma/coordinates/clustering/include'],
                  libraries=[lib_prefix+'theobald'],
                  library_dirs=[mdtraj.capi()['lib_dir']],
                  extra_compile_args=['-std=c99'])

    exts += [mle_trev_given_pi_dense_module,
             mle_trev_given_pi_sparse_module,
             mle_trev_sparse_module,
             regspatial_module,
             kmeans_module]

    if USE_CYTHON: # if we have cython available now, cythonize module
        exts = cythonize(exts)
    else:
        # replace pyx files by their pre generated c code.
        for e in exts:
            new_src = []
            for s in e.sources:
                new_src.append(s.replace('.pyx', '.c'))
            e.sources = new_src

    if openmp_enabled:
        warnings.warn('enabled openmp')
        omp_compiler_args = ['-fopenmp']
        omp_libraries = ['-lgomp'] if needs_gomp else []
        omp_defines = [('USE_OPENMP', None)]
        for e in exts:
            e.extra_compile_args += omp_compiler_args
            e.extra_link_args += omp_libraries
            e.define_macros += omp_defines

    return exts


def get_cmdclass():
    vervsioneer_cmds = versioneer.get_cmdclass()

    from distutils.command.build_ext import build_ext
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
            self.include_dirs = [get_include()]

    sdist_class = vervsioneer_cmds['sdist']
    class sdist(sdist_class):
        """ensure cython files are compiled to c, when distributing"""

        def run(self):
            # only run if .git is present
            if not os.path.exists('.git'):
                return

            try:
                from Cython.Build import cythonize
                print "cythonizing sources"
                cythonize(extensions())
            except ImportError:
                warnings.warn('sdist cythonize failed')
            return sdist_class.run(self)

    cmdclass = dict(build_ext=np_build,
                    sdist=sdist,
                    )

    vervsioneer_cmds.update(cmdclass)
    return vervsioneer_cmds


def script_entry_points():
    import pkgutil
    path = os.path.join('pyemma', 'cli')
    names = [name for _, name, _ in pkgutil.iter_modules([path])]
    s = ['%s = pyemma.cli.%s:main' % (script, script) for script in names]
    entry_points = {
        'console_scripts': s
    }

    return entry_points


metadata = dict(
    name='pyEMMA',
    maintainer='Martin K. Scherer',
    maintainer_email='m.scherer@fu-berlin.de',
    author='The Emma team',
    author_email='info@emma-project.org',
    url='http://github.com/markovmodel/PyEMMA',
    license='FreeBSD',
    description=DOCLINES[0],
    long_description=open('README.rst').read(),
    version=versioneer.get_version(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    keywords='Markov State Model Algorithms',
    # packages are found if their folder contains an __init__.py,
    packages=find_packages(),
    # install default emma.cfg into package.
    package_data=dict(pyemma=['pyemma.cfg']),
    entry_points=script_entry_points(),
    cmdclass=get_cmdclass(),
    tests_require=['nose'],
    test_suite='nose.collector',
    # runtime dependencies
    install_requires=['numpy>=1.6.0',
                      'scipy>=0.11',
                      'mdtraj',
                      'matplotlib'],
    zip_safe=False,
)

# this is only metadata and not used by setuptools
metadata['requires'] = ['numpy', 'scipy']

# not installing?
if len(sys.argv) == 1 or (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                          sys.argv[1] in ('--help-commands',
                                          '--version',
                                          'clean'))):
    pass
else:
    # setuptools>=2.2 can handle setup_requires
    metadata['setup_requires'] = ['numpy>=1.6.0',
                                  'setuptools>3.6',
                                  'mdtraj>=1.2.0',
                                  'nose',
                                 ]

    # when on git, we require cython
    if os.path.exists('.git'):
        warnings.warn('using git, require cython')
        metadata['setup_requires'] += ['cython>=0.22']

    # only require numpy and extensions in case of building/installing
    metadata['ext_modules'] = lazy_cythonize(extensions)

    # add argparse to runtime deps if python version is 2.6
    if sys.version_info[:2] == (2, 6):
        metadata['install_requires'] += ['argparse']

    # include ipython notebooks. Will be installed directly in site-packages
    #metadata['packages'] += ['pyemma-ipython']
    #metadata['include_package_data'] = True

try:
    setup(**metadata)
except VersionConflict as ve:
    print ve
    print "You need to manually upgrade your 'setuptools' installation!"
    " Please use these instructions to perform an upgrade and/or consult\n"
    " https://pypi.python.org/pypi/setuptools#installation-instructions"
    print getSetuptoolsError()
