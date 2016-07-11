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
_debug = True

_reg_np_handler()

_renamed_classes = {}
""" this dict performs a mapping between old and new names. A class can be renamed multiple times. """


class DeveloperError(Exception):
    """ the devs have done something wrong. """


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
    # TODO: this is the place to check for renamed classed and substitute it in the inp string.
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
        inter_map = OrderedDict(sorted(self._serialize_interpolation_map.iteritems()))

        # all version keys are integers
        if not all(is_int(k) for k in inter_map):
            raise DeveloperError("all keys of _serialize_interpolation_map have to be of type int.")

        # check mapping operations are contained in an iterable type
        if not all(isinstance(x, (list, tuple)) for x in inter_map.itervalues()):
            raise DeveloperError("all operations in _serialize_interpolation_map have "
                                 "to be contained in a list or tuple.")

        # check for valid operations: add, rm, mv
        valid_ops = ('set', 'rm', 'mv')
        if not all(action[0] in valid_ops for actions in inter_map.itervalues() for action in actions):
            raise DeveloperError("Your _serialize_interpolation_map contains invalid operations. "
                                 "Valid ops are: {}".format(valid_ops))

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
                    logger.debug("processing rule: %s" % str(a))
                if a[0] == 'set':
                    state[a[1]] = a[2]
                elif a[0] == 'mv':
                    try:
                        value = state.pop(a[1])
                        state[a[2]] = value
                    except KeyError:
                        raise DeveloperError("the previous version didn't "
                                             "store an attribute named '{}'".format(a[1]))
                elif a[0] == 'rm':
                    state.pop(a[1], None)
        if _debug:
            logger.debug("interpolated state: %s" % state)

    def _set_state_from_serializeable_fields_and_state(self, state, klass):
        """ set only fields from state, which are present in klass._serialize_fields """
        if _debug:
            logger.debug("restoring state for class %s" % klass)

        klass_version = state['_serialize_version']#state['__serialize_class_versions'][klass.__name__]
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
        res['__serialize_class_versions'] = {}
        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and klass._serialize_fields and not klass == SerializableMixIn:
                inc = self._get_state_of_serializeable_fields(klass)
                res.update(inc)


        res['_serialize_version'] = self.__class__.mro()[0]._serialize_version
        # for klass in self.__class__.mro():
        #     if hasattr(klass, '_serialize_version'):
        #         res['__serialize_class_versions'][klass.__name__] = klass._serialize_version

        if _debug:
            logger.debug("versions: %s" % res['__serialize_class_versions'])

        # handle special cases Estimator and Model, just use their parameters.
        if isinstance(self, Estimator):
            res.update(self.get_params())
            # remember if it has been estimated.
            res['_estimated'] = self._estimated
            try:
                res['model'] = self._model
            except AttributeError:
                pass

        if isinstance(self, Model):
            state = self.get_model_params()
            res.update(state)

        return res

    def __setstate__(self, state):
        from pyemma._base.estimator import Estimator
        from pyemma._base.model import Model

        classes_to_inspect = [c for c in self.__class__.mro() if hasattr(c, '_serialize_fields')
                              and c != SerializableMixIn and c != object and c != Estimator and c != Model]

        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and not klass == SerializableMixIn:
                self._set_state_from_serializeable_fields_and_state(state, klass=klass)

        if isinstance(self, Model):
            # remove items in state which are not model parameters
            names = self._get_model_param_names()
            new_state = {key: state[key] for key in names}

            self.update_model_params(**new_state)

        if isinstance(self, Estimator):
            self._estimated = state.pop('_estimated')
            model = state.pop('model', None)
            self._model = model

            # first set parameters of estimator, items in state which are not estimator parameters
            names = self._get_param_names()
            new_state = {key: state[key] for key in names if key in state}
            self.set_params(**new_state)
