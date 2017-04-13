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
from pyemma._base.serialization.jsonpickler_handlers import register_all_handlers as _reg_all_handlers
from pyemma._base.serialization.util import class_rename_registry
from pyemma._ext import jsonpickle
from pyemma._ext.jsonpickle.util import importable_name as _importable_name
from pyemma.util.types import is_int

logger = logging.getLogger(__name__)
_debug = False

if _debug:
    logger.level = logging.DEBUG

# indicate whether serialization handlers have already been registered
_handlers_registered = False


class DeveloperError(Exception):
    """ the devs have done something wrong. """


class OldVersionUnsupported(NotImplementedError):
    """ can not load recent models with old software versions. """


def load(file_like):
    """ loads a previously saved object of this class from a file.

    Parameter
    ---------
    file_like : str or file like object (has to provide read method).
        The file like object tried to be read for a serialized object.

    Returns
    -------
    obj : the de-serialized object
    """
    import bz2
    if not hasattr(file_like, 'read'):
        with contextlib.closing(bz2.BZ2File(file_like)) as fh:
            inp = fh.read()
    else:
        inp = bz2.decompress(file_like.read())

    if _debug:
        logger.debug("type of input: %s", type(inp))

    kw = {} if six.PY2 else {'encoding':'ascii'}
    inp = str(inp, **kw)

    inp = class_rename_registry.upgrade_old_names_in_json(inp)
    if _debug: pass
        #logger.debug("replaced {renamed} with {new}".format(renamed=renamed, new=new))

    if not _handlers_registered:
        _reg_all_handlers()

    obj = jsonpickle.loads(inp)

    return obj


class _SerializableBase(object):
    """ Base class of serializable classes.

       Derive from this class to make your class serializable via save and load methods.
    """

    _serialize_fields = ()
    """ attribute names to serialize """

    def save(self, filename_or_file, compression_level=9, save_streaming_chain=False):
        """
        Parameters
        -----------
        filename_or_file: str or file like
            path to desired output file or a type which implements the file protocol (accepting bytes as input).
        compression_level : int
            if given, must be a number between 1 and 9.
        save_streaming_chain : boolean, default=False
            if True, the data_producer(s) of this object will also be saved in the given file.
            
        Examples
        --------
        TODO: write me
        """
        if not _handlers_registered:
            _reg_all_handlers()
        # if we are serializing a pipeline element, store whether to store the chain elements.
        old_flag = self._save_data_producer
        self._save_data_producer = save_streaming_chain
        assert self._save_data_producer == save_streaming_chain
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            if isinstance(self, Loggable):
                self.logger.exception('During saving the object ("{error}") '
                                      'the following error occurred'.format(error=e))
            raise
        finally:
            # restore old state.
            self._save_data_producer = old_flag

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
            raise DeveloperError("your class does not implement the serialization protocol of PyEMMA.")

        return obj

    @property
    def _save_data_producer(self):
        if not hasattr(self, '_SerializableBase__save_data_producer'):
            self.__save_data_producer = False
        return self.__save_data_producer

    @_save_data_producer.setter
    def _save_data_producer(self, value):
        self.__save_data_producer = value
        # forward flag to the next data producer
        if hasattr(self, 'data_producer') and self.data_producer and self.data_producer is not self:
            assert isinstance(self.data_producer, _SerializableBase), self.data_producer
            self.data_producer._save_data_producer = value


class SerializableMixIn(_SerializableBase):
    """ Base class of serializable classes using get/set_state.

    Derive from this class to make your class serializable. Do not forget to
    add a version number to your class to distinguish old and new copies of the
    source code. The static attribute '_serialize_fields' is a iterable of names,
    which are preserved during serialization.
    
    To aid the process of loading old models in a new version of the software, there
    is the the static field '_serialize_interpolation_map', which is a mapping from
    old version number to a set of operations to transform the old class state to the
    recent version of the class.
    
    Valid operations are:
    1. ('rm', 'name') -> delete the attribute with given name.
    2. ('mv', 'old', 'new') -> rename the attribute from 'old' to 'new'.
    3. ('set', 'name', value) -> set an attribute with name 'name' to given value.
    4. ('map', 'name', func) -> apply the function 'func' to attribute 'name'. The function
      should accept one argument, namely the attribute and return the new value for it.
    
    Similar to map, there are two callbacks to hook into the serialization process:
    5. ('set_state_hook', func) -> a function which may transform the state dictionary
       before __getstate__ returns.

    Example
    -------

    >>> import pyemma
    >>> from io import BytesIO
    >>> class MyClass(SerializableMixIn):
    ...    _serialize_version = 0
    ...    _serialize_fields = ['x']
    ...    def __init__(self, x=42):
    ...        self.x = x

    >>> inst = MyClass()
    >>> file = BytesIO()
    >>> inst.save(file)
    >>> _= file.seek(0)
    >>> inst_restored = pyemma.load(file)
    >>> assert inst_restored.x == inst.x # doctest: +SKIP
    # skipped because MyClass is not importable.

    """

    _serialize_fields = ()
    """ attribute names to serialize """

    def _get_state_of_serializeable_fields(self, klass):
        """ :return a dictionary {k:v} for k in self.serialize_fields and v=getattr(self, k)"""
        res = {}
        assert all(isinstance(f, six.string_types) for f in klass._serialize_fields)
        for field in klass._serialize_fields:
            # only try to get fields, we actually have.
            if hasattr(self, field):
                res[field] = getattr(self, field)
        return res

    def _validate_interpolation_map(self, klass):
        # version numbers should be sorted
        from collections import OrderedDict
        inter_map = OrderedDict(sorted(klass._serialize_interpolation_map.items()))
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

        klass._serialize_interpolation_map = inter_map

    def __interpolate(self, state, klass):
        # First lookup the version of klass in the state (this maps from old versions too).
        # Lookup attributes in interpolation map according to version number of the class.
        # Drag in all prior versions attributes
        if not hasattr(klass, '_serialize_interpolation_map'):
            return

        klass_version = self._get_version_for_class_from_state(state, klass)

        if klass_version > klass._serialize_version:
            return

        self._validate_interpolation_map(klass)

        if _debug:
            logger.debug("input state: %s" % state)
        for key in klass._serialize_interpolation_map.keys():
            if not (klass._serialize_version > key >= klass_version):
                if _debug:
                    logger.debug("skipped interpolation rules for version %s" % key)
                continue
            if _debug:
                logger.debug("processing rules for version %s" % key)
            actions = klass._serialize_interpolation_map[key]
            for a in actions:
                if _debug:
                    logger.debug("processing rule: %s", str(a))
                if len(a) == 3:
                    operation, name, value = a
                    if operation == 'set':
                        state[name] = value
                    elif operation == 'mv':
                        try:
                            arg = state.pop(name)
                            state[value] = arg
                        except KeyError:
                            raise DeveloperError("the previous version didn't "
                                                 "store an attribute named '{}'".format(a[1]))
                    elif operation == 'map':
                        func = value
                        if hasattr(func, '__func__'):
                            func = func.__func__
                        assert callable(func)
                        state[name] = func(state[name])
                elif len(a) == 2:
                    action, value = a
                    if action == 'rm':
                        state.pop(value, None)
        if _debug:
            logger.debug("interpolated state: %s", state)

    def _get_version_for_class_from_state(self, state, klass):
        """ retrieves the version of the current klass from the state mapping from old locations to new ones. """

        """ klass may have renamed, so we have to look this up in _new_to_old.
        
        """

        names = [_importable_name(klass)]
        # lookup old names, handled by current klass.
        names.extend(class_rename_registry.old_handled_by(klass))
        for n in names:
            try:
                return state['class_tree_versions'][n]
            except KeyError:
                continue
        # if we did not find a suitable version number return infinity.
        return float('inf')

    def _set_state_from_serializeable_fields_and_state(self, state, klass):
        """ set only fields from state, which are present in klass._serialize_fields """
        if _debug:
            logger.debug("restoring state for class %s", klass)

        klass.__interpolate(self, state, klass)

        for field in klass._serialize_fields:
            if field in state:
                setattr(self, field, state[field])
            else:
                if _debug:
                    logger.debug("skipped %s, because it is not declared in _serialize_fields", field)

    def __getstate__(self):
        # We just dump the version number for comparison with the actual class.
        # Note: we do not want to set the version number in __setstate__,
        # since we obtain it from the actual definition.
        if not hasattr(self, '_serialize_version'):
            raise DeveloperError('The "{klass}" should define a static "_serialize_version" attribute.'
                                 .format(klass=self.__class__))

        from pyemma._base.estimator import Estimator
        from pyemma._base.model import Model

        #res = {'_serialize_version': self._serialize_version,}
        # TODO: do we really need to store fields here?
        #'_serialize_fields': self._serialize_fields}
        res = {'class_tree_versions': {}}
        for c in self.__class__.mro():
            name = _importable_name(c)
            if hasattr(c, '_serialize_version'):
                v = c._serialize_version
            else:
                v = -1
            res['class_tree_versions'][name] = v

        # if we want to save the chain, do this now:
        if self._save_data_producer:
            assert hasattr(self, 'data_producer')
            res['data_producer'] = dp = self.data_producer

        classes_to_inspect = [c for c in self.__class__.mro() if hasattr(c, '_serialize_fields')
                              and c != SerializableMixIn and c != object and c != Estimator and c != Model]
        if _debug:
            logger.debug("classes to inspect during setstate: \n%s" % classes_to_inspect)
        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and klass._serialize_fields and not klass == SerializableMixIn:
                inc = self._get_state_of_serializeable_fields(klass)
                res.update(inc)

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
        # no backward compatibility.
        # TODO: is the child impl version sufficient to check this? Deleted classes could also trigger this...
        #if self._get_version_for_class_from_state(state, klass=type(self)) > self._serialize_version:
        #    raise OldVersionUnsupported("Can not load recent models with old version of PyEMMA."
        #                                " You need at least {supported}".format(supported=state['_pyemma_version']))

        from pyemma._base.estimator import Estimator
        from pyemma._base.model import Model

        classes_to_inspect = [c for c in self.__class__.mro() if hasattr(c, '_serialize_fields')
                              and c != SerializableMixIn and c != object and c != Estimator and c != Model]

        for klass in classes_to_inspect:
            if hasattr(klass, '_serialize_fields') and klass._serialize_fields and hasattr(klass, '_serialize_version'):
                self._set_state_from_serializeable_fields_and_state(state, klass=klass)

        if hasattr(self, 'set_model_params') and hasattr(self, '_get_model_param_names'):
            # only apply params suitable for the current model
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

        if hasattr(self, 'data_producer') and 'data_producer' in state:
            self.data_producer = state['data_producer']

        if hasattr(state, '_pyemma_version'):
            self._pyemma_version = state['_pyemma_version']
