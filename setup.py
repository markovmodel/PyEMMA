#!/usr/bin/env python

# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""EMMA: Emma's Markov Model Algorithms

EMMA is an open source collection of algorithms implemented mostly in
`NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org>`_
to analyze trajectories generated from any kind of simulation
(e.g. molecular trajectories) via Markov state models (MSM).

"""

from __future__ import print_function, absolute_import

import sys
import os
import versioneer
import warnings
from io import open

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: MacOS X
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Operating System :: Microsoft :: Windows
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
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
    print(getSetuptoolsError())
    sys.exit(23)

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
    from numpy import get_include as _np_inc
    np_inc = _np_inc()

    exts = []

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
                      np_inc,
                      'pyemma/coordinates/clustering/include',
                  ],
                  libraries=[lib_prefix+'theobald'],
                  library_dirs=[mdtraj.capi()['lib_dir']],
                  extra_compile_args=['-std=c99', '-g', '-O3', '-pg'])
    kmeans_module = \
        Extension('pyemma.coordinates.clustering.kmeans_clustering',
                  sources=[
                      'pyemma/coordinates/clustering/src/kmeans.c',
                      'pyemma/coordinates/clustering/src/clustering.c'],
                  include_dirs=[
                      mdtraj.capi()['include_dir'],
                      np_inc,
                      'pyemma/coordinates/clustering/include'],
                  libraries=[lib_prefix+'theobald'],
                  library_dirs=[mdtraj.capi()['lib_dir']],
                  extra_compile_args=['-std=c99'])

    covar_module = \
        Extension('pyemma.coordinates.estimators.covar.covar_c.covartools',
                  sources=['pyemma/coordinates/estimators/covar/covar_c/covartools.pyx',
                           'pyemma/coordinates/estimators/covar/covar_c/_covartools.c'],
                  include_dirs=['pyemma/coordinates/estimators/covar/covar_c/',
                                np_inc,
                                ],
                  extra_compile_args=['-std=c99', '-O3'])

    exts += [regspatial_module,
             kmeans_module,
             covar_module,
             ]

    if not USE_CYTHON:
        # replace pyx files by their pre generated c code.
        for e in exts:
            new_src = []
            for s in e.sources:
                new_src.append(s.replace('.pyx', '.c'))
            e.sources = new_src
    else:
        exts = cythonize(exts)

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
    versioneer_cmds = versioneer.get_cmdclass()

    sdist_class = versioneer_cmds['sdist']
    class sdist(sdist_class):
        """ensure cython files are compiled to c, when distributing"""

        def run(self):
            # only run if .git is present
            if not os.path.exists('.git'):
                print("Not on git, can not create source distribution")
                return

            try:
                from Cython.Build import cythonize
                print("cythonizing sources")
                cythonize(extensions())
            except ImportError:
                warnings.warn('sdist cythonize failed')
            return sdist_class.run(self)

    versioneer_cmds['sdist'] = sdist

    return versioneer_cmds


metadata = dict(
    name='pyEMMA',
    maintainer='Martin K. Scherer',
    maintainer_email='m.scherer@fu-berlin.de',
    author='The Emma team',
    author_email='info@emma-project.org',
    url='http://github.com/markovmodel/PyEMMA',
    license='LGPLv3+',
    description=DOCLINES[0],
    long_description=open('README.rst', encoding='utf8').read(),
    version=versioneer.get_version(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    keywords='Markov State Model Algorithms',
    # packages are found if their folder contains an __init__.py,
    packages=find_packages(),
    # install default emma.cfg into package.
    package_data=dict(pyemma=['pyemma.cfg']),
    cmdclass=get_cmdclass(),
    tests_require=['nose'],
    test_suite='nose.collector',
    # runtime dependencies
    install_requires=['numpy>=1.7.0',
                      'scipy>=0.11',
                      'mdtraj>=1.7.0',
                      'matplotlib',
                      'msmtools>=1.1.3',
                      'thermotools>=0.2.3',
                      'bhmm>=0.6,<0.7',
                      'joblib>0.8.4',
                      'pyyaml',
                      'psutil>=3.1.1',
                      'decorator>=4.0.0',
                      'progress-reporter',
                      ],
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
    metadata['setup_requires'] = ['numpy>=1.7.0',
                                  'mdtraj>=1.7.0',
                                  'nose',
                                  ]
    if sys.version_info.major == 2:
        # kick it since causes headages with conda recently...
        #metadata['install_requires'] += ['mock']
        pass

    metadata['package_data'] = {
                                'pyemma': ['pyemma.cfg', 'logging.yml'],
                                'pyemma.coordinates.tests': ['data/*'],
                                'pyemma.datasets': ['*.npz'],
                                'pyemma.util.tests': ['data/*'],
                                }

    # when on git, we require cython
    if os.path.exists('.git'):
        warnings.warn('using git, require cython')
        metadata['setup_requires'] += ['cython>=0.22']

    # only require numpy and extensions in case of building/installing
    metadata['ext_modules'] = lazy_cythonize(callback=extensions)

setup(**metadata)

