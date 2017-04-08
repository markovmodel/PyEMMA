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

import contextlib
import logging

import six

from pyemma._base.logging import Loggable
from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler as _reg_np_handler
from pyemma._ext import jsonpickle
from pyemma.util.types import is_string, is_int

logger = logging.getLogger(__name__)
_debug = False

if _debug:
    logger.level = logging.DEBUG

_reg_np_handler()

_renamed_classes = {}
""" this dict performs a mapping between old and new names. A class can be renamed multiple times. """


class DeveloperError(Exception):
    """ the devs have done something wrong. """


def load(file_like):
    import bz2
    with contextlib.closing(bz2.BZ2File(file_like)) as fh:
        inp = fh.read()
    kw = {}
    if six.PY3:
        kw['encoding'] = 'ascii'
    inp = str(inp, **kw)

    for renamed in _renamed_classes:
        new = _renamed_classes[renamed]
        inp = inp.replace('"%s"' % renamed, new)
        if _debug:
            logger.debug("replacing {renamed} with {new}".format(renamed=renamed, new=new))
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
    ...    _serialize_fields = ['x']
    ...    def __init__(self, x=42):
    ...        self.x = x

    >>> inst = MyClass()
    >>> file = tempfile.NamedTemporaryFile()
    >>> inst.save(file.name)
    >>> inst_restored = pyemma.load(file.name)
    >>> assert inst_restored.x == inst.x # doctest: +SKIP


    """

    _serialize_fields = ()
    """ attribute names to serialize """

    def save(self, filename_or_file, compression_level=9):
        """
        Parameters
        -----------
        filename_or_file: str or file like
            path to desired output file or a type which implements the file protocol (accepting bytes as input).
        """
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            if isinstance(self, Loggable):
                self.logger.exception('During saving the object ("{error}") '
                                      'the following error occurred'.format(error=e))
            raise

        if six.PY3:
            flattened = bytes(flattened, encoding='ascii')

        import bz2
        compressed = bz2.compress(flattened, compresslevel=compression_level)
        if not hasattr(filename_or_file, 'write'):
            with open(filename_or_file, mode='wb') as fh:
                fh.write(compressed)
        else:
            filename_or_file.write(compressed)
            filename_or_file.flush()

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

    def _validate_interpolation_map(self):
        # version numbers should be sorted
        from collections import OrderedDict
        inter_map = OrderedDict(sorted(self._serialize_interpolation_map.items()))
        if _debug:
            logger.debug("validate map: %s", inter_map)

        # check for valid operations: add, rm, mv, map
        valid_ops = ('set', 'rm', 'mv', 'map')
        for k, v in inter_map.items():
            if not is_int(k):
                raise DeveloperError("all keys of _serialize_interpolation_map "
                                     "have to be of type int (class version numbers)")
            if not isinstance(v, (list, tuple)):
                raise DeveloperError("actions per version have to be list or tuple")

            for action in v:
                if action[0] not in valid_ops:
                    raise DeveloperError("Your _serialize_interpolation_map contains invalid operations. "
                                         "Valid ops are: {valid_ops}. You provided {provided}"
                                         .format(valid_ops=valid_ops, provided=action[0]))

        self._serialize_interpolation_map = inter_map

    def __interpolate(self, state):
        # Lookup attributes in interpolation map according to version number of the class.
        # Drag in all prior versions attributes
        self._validate_interpolation_map()

        logger.debug("input state: %s" % state)
        state_version = state['_serialize_version']
        for key in self._serialize_interpolation_map.keys():
            if not (self._serialize_version > key >= state_version):
                if _debug:
                    logger.debug("skipped interpolation rules for version %s" % key)
                continue
            if _debug:
                logger.debug("processing rules for version %s" % key)
            actions = self._serialize_interpolation_map[key]
            for a in actions:
                if _debug:
                    logger.debug("processing rule: %s", str(a))
                if len(a) == 3:
                    operation, name, value = a
                    if operation == 'set':
                        state[name] = value
                    elif operation == 'mv':
                        try:
                            value = state.pop(a[1])
                            state[a[2]] = value
                        except KeyError:
                            raise DeveloperError("the previous version didn't "
                                                 "store an attribute named '{}'".format(a[1]))
                elif len(a) == 2:
                    action, value = a
                    if action == 'rm':
                        state.pop(value, None)
        if _debug:
            logger.debug("interpolated state: %s", state)

    def _set_state_from_serializeable_fields_and_state(self, state, klass):
        """ set only fields from state, which are present in klass._serialize_fields """
        if _debug:
            logger.debug("restoring state for class %s" % klass)

        klass_version = state['_serialize_version']  # state['__serialize_class_versions'][klass.__name__]
        if klass_version < klass._serialize_version and hasattr(self, '_serialize_interpolation_map'):
            self.__interpolate(state)

        for field in klass._serialize_fields:
            if field in state:
                setattr(self, field, state[field])
            else:
                if _debug:
                    logger.debug("skipped %s, because it is not declared in _serialize_fields" % field)

    def __getstate__(self):
        # We just dump the version number for comparison with the actual class.
        # Note: we do not want to set the version number in __setstate__,
        # since we obtain it from the actual definition.
        if not hasattr(self, '_serialize_version'):
            raise DeveloperError('The "{klass}" should define a static "_serialize_version" attribute.'
                                 .format(klass=self.__class__))

        from pyemma._base.estimator import Estimator
        from pyemma._base.model import Model

        res = {'_serialize_version': self._serialize_version,
               # TODO: do we really need to store fields here?
               '_serialize_fields': self._serialize_fields}

        classes_to_inspect = [c for c in self.__class__.mro() if hasattr(c, '_serialize_fields')
                              and c != SerializableMixIn and c != object and c != Estimator and c != Model]
        if _debug:
            logger.debug("classes to inspect during setstate: \n%s" % classes_to_inspect)
        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and klass._serialize_fields and not klass == SerializableMixIn:
                inc = self._get_state_of_serializeable_fields(klass)
                res.update(inc)

        # We only store the version of the most specialized class
        res['_serialize_version'] = self.__class__.mro()[0]._serialize_version
        assert res['_serialize_version'] == self._serialize_version

        # handle special cases Estimator and Model, just use their parameters.
        if hasattr(self, 'get_params'):
            res.update(self.get_params())
            # remember if it has been estimated.
            res['_estimated'] = self._estimated
            try:
                res['model'] = self._model
            except AttributeError:
                pass

        if hasattr(self, 'get_model_params'):
            state = self.get_model_params()
            res.update(state)

        # store the current software version
        from pyemma import version
        res['_pyemma_version'] = version

        return res

    def __setstate__(self, state):
        from pyemma._base.estimator import Estimator
        from pyemma._base.model import Model

        classes_to_inspect = [c for c in self.__class__.mro() if hasattr(c, '_serialize_fields')
                              and c != SerializableMixIn and c != object and c != Estimator and c != Model]

        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and klass._serialize_fields and hasattr(klass, '_serialize_version'):
                self._set_state_from_serializeable_fields_and_state(state, klass=klass)

        if hasattr(self, 'set_model_params') and hasattr(self, '_get_model_param_names'):
            names = self._get_model_param_names()
            new_state = {key: state[key] for key in names}

            self.set_model_params(**new_state)

        if hasattr(self, 'set_params') and hasattr(self, '_get_param_names'):
            self._estimated = state.pop('_estimated')
            model = state.pop('model', None)
            self._model = model

            # first set parameters of estimator, items in state which are not estimator parameters
            names = self._get_param_names()
            new_state = {key: state[key] for key in names if key in state}
            self.set_params(**new_state)

        if hasattr(state, '_pyemma_version'):
            self._pyemma_version = state['_pyemma_version']
