# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pyemma._ext import jsonpickle
from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler as _reg_np_handler
from pyemma._base.logging import Loggable
from pyemma.util.types import is_string
import contextlib
import six

_reg_np_handler()


class LoadedObjectVersionMismatchException(Exception):
    """ the version does not match the current version of the library. """


class DeveloperError(Exception):
    """ the devs has done something wrong. """


def load(file_like):
    import bz2
    if is_string(file_like):
        file_like = bz2.BZ2File(file_like)
    with contextlib.closing(file_like) as file_like:
        inp = file_like.read()
    kw = {}
    if six.PY3:
        kw['encoding'] = 'ascii'
    inp = str(inp, **kw)
    obj = jsonpickle.loads(inp)

    return obj


class SerializableMixIn(object):
    """

    """

    def save(self, filename):
        """
        Parameters
        -----------
        filename: str or file like
            path to desired output file or a type which implements the file protocol.
        """
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            if isinstance(self, Loggable):
                self.logger.exception('During saving the object ("%s") '
                                      'the following error occured' % e)
            raise

        if six.PY3:
            flattened = bytes(flattened, encoding='ascii')

        import bz2
        with contextlib.closing(bz2.BZ2File(filename, 'w')) as fh:
            try:
                fh.write(flattened)
            except:
                raise

    @classmethod
    def load(cls, file_like):
        """ loads a previously saved object of this class from a file.

        Parameter
        ---------
        file_like : str or file like object (has to provide read method).
            The file like object tried to be read for a serialized object.

        Returns
        -------
        obj : the de-serialized object
        """
        obj = load(file_like)

        if obj.__class__ != cls:
            raise ValueError("Given file '%s' did not contain the right type:"
                             " desired(%s) vs. actual(%s)" % (file_like, cls, obj.__class__))
        if not hasattr(cls, '_version'):
            raise DeveloperError("your class does not implement the deserialization protocol of PyEMMA.")

        if obj._version != cls._version:
            pass
            #raise LoadedObjectVersionMismatchException("Version mismatch")

        return obj
