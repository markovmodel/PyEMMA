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
    """ Base class of serializable classes.

    Derive from this class to make your class serializable. Do not forget to
    add a version number to your class to distinguish old and new copies of the
    source code:

    >>> import tempfile, pyemma, os
    >>> class MyClass(SerializableMixIn):
    ...    _serialize_version = 0
    ...    def __init__(self, x=42):
    ...        self.x = x

    >>> inst = MyClass()
    >>> file = tempfile.NamedTemporaryFile()
    >>> inst.save(file.name)
    >>> inst_restored = pyemma.load(file.name)
    >>> assert inst_restored.x == inst.x
    >>> os.unlink(file.name)

    In case

    """

    _serialize_fields = ()
    """ attribute names to serialize """

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
        if not hasattr(cls, '_serialize_version'):
            raise DeveloperError("your class does not implement the deserialization protocol of PyEMMA.")

        if obj._version != cls._serialize_version:
            raise LoadedObjectVersionMismatchException("Version mismatch")

        return obj

    def _get_state_of_serializeable_fields(self, klass):
        """ :return a dictionary {k:v} for k in self.serialize_fields and v=getattr(self, k)"""
        if not hasattr(self, '_serialize_fields'):
            return {}
        res = {}
        for field in klass._serialize_fields:
            # only try to get fields, we actually have.
            if hasattr(self, field):
                res[field] = getattr(self, field)
        return res

    def _set_state_from_serializeable_fields_and_state(self, state, klass):
        """ set only fields from state, which are present in klass._serialize_fields """
        for field in klass._serialize_fields:
            if field in state:
                setattr(self, field, state[field])

    def __getstate__(self):
        return {'_serialize_version': self._version,
                '_serialize_fields': self._serialize_fields}

    def __setstate__(self, state):
        self._version = state.pop('_serialize_version')
        self._serialize_fields = state.pop('_serialize_fields', ())

