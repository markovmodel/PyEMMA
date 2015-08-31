
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
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


'''
Created on 17.02.2014

@author: marscher
'''
from __future__ import absolute_import, print_function

import os
import errno
import tempfile
import shutil


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

    Examples
    --------
    >>> import os
    >>> with TemporaryDirectory() as tmp:
    ...    path = os.path.join(tmp, "myfile.dat")
    ...    fh = open(path, 'w')
    ...    _ = fh.write('hello world')
    ...    fh.close()

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, prefix='', suffix='', dir=None):
        self.prefix = prefix
        self.suffix = suffix
        self.dir = dir
        self.tmpdir = None

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix,
                                       dir=self.dir)
        return self.tmpdir

    def __exit__(self, *args):
        shutil.rmtree(self.tmpdir, ignore_errors=True)