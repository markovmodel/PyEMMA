
# This file is part of MSMTools.
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

"""
utility functions for python setup
"""
import tempfile
import os
import sys
import shutil
import warnings
import setuptools
import contextlib


@contextlib.contextmanager
def TemporaryDirectory():
    n = tempfile.mkdtemp()
    yield n
    shutil.rmtree(n)


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename, fake=False):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """
    if fake:
        yield
        return
    oldstdchannel = dest_file = None
    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


# From http://stackoverflow.com/questions/
# 7018879/disabling-output-when-compiling-with-distutils
def has_function(compiler, funcname, headers):
    if not isinstance(headers, (tuple, list)):
        headers = [headers]
    with TemporaryDirectory() as tmpdir, stdchannel_redirected(sys.stderr, os.devnull), \
             stdchannel_redirected(sys.stdout, os.devnull):
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            for h in headers:
                f.write('#include <%s>\n' % h)
            f.write('int main(void) {\n')
            f.write(' %s();\n' % funcname)
            f.write('return 0;}')
            f.close()
            objects = compiler.compile([fname], output_dir=tmpdir)
            compiler.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except (setuptools.distutils.errors.CompileError, setuptools.distutils.errors.LinkError):
            return False
        except:
            import traceback
            traceback.print_last()
            return False
        return True


def detect_openmp(compiler):
    from distutils.log import debug
    from copy import deepcopy
    compiler = deepcopy(compiler) # avoid side-effects
    has_openmp = has_function(compiler, 'omp_get_num_threads', headers='omp.h')
    debug('[OpenMP] compiler %s has builtin support', compiler)
    additional_libs = []
    if not has_openmp:
        debug('[OpenMP] compiler %s needs library support', compiler)
        if sys.platform == 'darwin':
            compiler.add_library('iomp5')
        elif sys.platform.startswith('linux'):
            compiler.add_library('gomp')
        has_openmp = has_function(compiler, 'omp_get_num_threads', headers='omp.h')
        if has_openmp:
            additional_libs = [compiler.libraries[-1]]
            debug('[OpenMP] added library %s', additional_libs)
    return has_openmp, additional_libs


# has_flag and cpp_flag taken from https://github.com/pybind/python_example/blob/master/setup.py
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with TemporaryDirectory() as tmpdir, \
            stdchannel_redirected(sys.stderr, os.devnull), \
            stdchannel_redirected(sys.stdout, os.devnull):
        f = tempfile.mktemp(suffix='.cpp', dir=tmpdir)
        with open(f, 'w') as fh:
            fh.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f], extra_postargs=[flagname], output_dir=tmpdir)
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler ({})-- at least C++11 support '
                           'is needed!'.format(compiler))


def parse_setuppy_commands():
    """Check the commands and respond appropriately.
    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).

    Adopted from scipy setup
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg',
                     'build_sphinx')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        return False

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating Cython sources and expanding templates".format(
                  ' '.join(sys.argv[1:])))

    return True

