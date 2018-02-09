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

import logging

from pyemma._base.loggable import Loggable
from pyemma._base.serialization.util import _importable_name

logger = logging.getLogger(__name__)
_debug = False

if _debug:
    logger.level = logging.DEBUG


class ClassVersionException(Exception):
    """ the devs have done something wrong. """


class OldVersionUnsupported(NotImplementedError):
    """ can not load recent models with old software versions. """


class IntegrityError(Exception):
    """ data mismatches of stored model parameters and pickled data. """


class Modifications(object):
    # class to create state modifications used to handle refactored classes.

    def __init__(self):
        self._ops = []

    def rm(self, name):
        self._ops.append(('rm', name))
        return self

    def mv(self, name, new_name):
        self._ops.append(('mv', name, new_name))
        return self

    def map(self, name, callable):
        self._ops.append(('map', name, callable))
        return self

    def set(self, name, value):
        self._ops.append(('set', name, value))
        return self

    def list(self):
        return self._ops

    @staticmethod
    def apply(modifications, state):
        """ applies modifications to given state
        Parameters
        ----------
        modifications: list of tuples
            created by this class.list method.
        state: dict
            state dictionary
        """
        count = 0
        for a in modifications:
            if _debug:
                assert a[0] in ('set', 'mv', 'map', 'rm')
                logger.debug("processing rule: %s", str(a))
            if len(a) == 3:
                operation, name, value = a
                if operation == 'set':
                    state[name] = value
                    count += 1
                elif operation == 'mv':
                    try:
                        arg = state.pop(name)
                        state[value] = arg
                        count += 1
                    except KeyError:
                        raise ClassVersionException("the previous version didn't "
                                                    "store an attribute named '{}'".format(a[1]))
                elif operation == 'map':
                    func = value
                    if hasattr(func, '__func__'):
                        func = func.__func__
                    assert callable(func)
                    state[name] = func(state[name])
                    count += 1
            elif len(a) == 2:
                action, value = a
                if action == 'rm':
                    state.pop(value, None)
                    count += 1
        assert count == len(modifications), 'was not able to process all modifications on state'


class SerializableMixIn(object):
    """ Base class of serializable classes using get/set_state.

    Derive from this class to make your class serializable. Do not forget to
    add a version number to your class to distinguish old and new copies of the
    source code. The static attribute '__serialize_fields' is a iterable of names,
    which are preserved during serialization.

    To aid the process of loading old models in a new version of the software, there
    is the the static field '__serialize_interpolation_map', which is a mapping from
    old version number to a set of operations to transform the old class state to the
    recent version of the class.

    Valid operations are:
    1. ('rm', 'name') -> delete the attribute with given name.
    2. ('mv', 'old', 'new') -> rename the attribute from 'old' to 'new'.
    3. ('set', 'name', value) -> set an attribute with name 'name' to given value.
    4. ('map', 'name', func) -> apply the function 'func' to attribute 'name'. The function
      should accept one argument, namely the attribute and return the new value for it.

    Example
    -------

    >>> import pyemma
    >>> from pyemma.util.contexts import named_temporary_file
    >>> class MyClass(SerializableMixIn):
    ...    __serialize_version = 0
    ...    __serialize_fields = ['x']
    ...    def __init__(self, x=42):
    ...        self.x = x

    >>> inst = MyClass()
    >>> with named_temporary_file() as file: # doctest: +SKIP
    ...    inst.save(file) # doctest: +SKIP
    ...    inst_restored = pyemma.load(file) # doctest: +SKIP
    >>> assert inst_restored.x == inst.x # doctest: +SKIP
    # skipped because MyClass is not importable.
    """
    __serialize_version = None
    """ version of class definition """

    __serialize_fields = ()
    """ attribute names to serialize """

    __serialize_modifications_map = {}

    def __new__(cls, *args, **kwargs):
        assert cls != SerializableMixIn.__class__  # don't allow direct instances of this class.
        SerializableMixIn._get_version(cls)
        res = super(SerializableMixIn, cls).__new__(cls)
        return res

    @staticmethod
    def _get_version(cls, require=True):
        name = cls.__name__
        if name.startswith('_'):
            name = name[1:]
        attr = '_%s__serialize_version' % name
        version = getattr(cls, attr, None)
        if require:
            if version is None:
                raise ClassVersionException('{} does not have the private field __serialize_version'.format(cls))
            if not isinstance(version, int):
                raise ClassVersionException('{} does not have an integer __serialize_version'.format(cls))
        # check for int
        return version

    @staticmethod
    def _get_serialize_fields(cls):
        name = cls.__name__
        if name.startswith('_'):
            name = name[1:]
        attr = '_%s__serialize_fields' % name
        fields = getattr(cls, attr, ())
        assert all(isinstance(f, str) for f in fields)
        return fields

    @staticmethod
    def _get_interpolation_map(cls):
        name = cls.__name__
        if name.startswith('_'):
            name = name[1:]
        attr = '_%s__serialize_modifications_map' % name
        map = getattr(cls, attr, {})
        return map

    def save(self, file_name, model_name='default', overwrite=False, save_streaming_chain=False):
        r""" saves the current state of this object to given file and name.

        Parameters
        -----------
        file_name: str
            path to desired output file
        model_name: str, default='default'
            creates a group named 'model_name' in the given file, which will contain all of the data.
            If the name already exists, and overwrite is False (default) will raise a RuntimeError.
        overwrite: bool, default=False
            Should overwrite existing model names?
        save_streaming_chain : boolean, default=False
            if True, the data_producer(s) of this object will also be saved in the given file.

        Examples
        --------
        >>> import pyemma, numpy as np
        >>> from pyemma.util.contexts import named_temporary_file
        >>> m = pyemma.msm.MSM(P=np.array([[0.1, 0.9], [0.9, 0.1]]))

        >>> with named_temporary_file() as file: # doctest: +SKIP
        ...    m.save(file, 'simple') # doctest: +SKIP
        ...    inst_restored = pyemma.load(file, 'simple') # doctest: +SKIP
        >>> np.testing.assert_equal(m.P, inst_restored.P) # doctest: +SKIP
        """
        import six
        if six.PY2:
            raise NotImplementedError('This feature is only available on Python3. Consider upgrading.')
        from pyemma._base.serialization.h5file import H5File
        try:
            with H5File(file_name=file_name) as f:
                f.add_serializable(model_name, obj=self, overwrite=overwrite, save_streaming_chain=save_streaming_chain)
        except Exception as e:
            msg = ('During saving the object {obj}") '
                   'the following error occurred: {error}'.format(obj=self, error=e))
            if isinstance(self, Loggable):
                self.logger.exception(msg)
            else:
                logger.exception(msg)
            raise

    @classmethod
    def load(cls, file_name, model_name='default'):
        """ loads a previously saved object of this class from a file.

        Parameters
        ----------
        file_name : str or file like object (has to provide read method).
            The file like object tried to be read for a serialized object.
        model_name: str, default='default'
            if multiple models are contained in the file, these can be accessed by
            their name. Use func:`pyemma.list_models` to get a representation of all stored models.

        Returns
        -------
        obj : the de-serialized object
        """
        import six
        if six.PY2:
            raise NotImplementedError('This feature is only available on Python3. Consider upgrading.')
        from .h5file import H5File
        with H5File(file_name, model_name=model_name, mode='r') as f:
            return f.model

    @property
    def _save_data_producer(self):
        try:
            return self.__save_data_producer
        except AttributeError:
            self.__save_data_producer = False
        return self.__save_data_producer

    @_save_data_producer.setter
    def _save_data_producer(self, value):
        self.__save_data_producer = value
        # forward flag to the next data producer
        if (value and
                hasattr(self, 'data_producer') and self.data_producer and self.data_producer is not self):
            # ensure the data_producer is serializable
            if not hasattr(self.data_producer.__class__, '_serialize_version'):
                raise RuntimeError('class in chain is not serializable: {}'.format(self.data_producer.__class__))
            self.data_producer._save_data_producer = value

    def _get_state_of_serializeable_fields(self, klass, state):
        """ :return a dictionary {k:v} for k in self.serialize_fields and v=getattr(self, k)"""
        for field in SerializableMixIn._get_serialize_fields(klass):
            # only try to get fields, we actually have.
            if hasattr(self, field):
                if _debug and field in state:
                    logger.debug('field "%s" already in state!', field)
                state[field] = getattr(self, field)
        return state

    @staticmethod
    def __interpolate(state, klass):
        # First lookup the version of klass in the state (this maps from old versions too).
        # Lookup attributes in interpolation map according to version number of the class.
        # Drag in all prior versions attributes
        modifications_map = SerializableMixIn._get_interpolation_map(klass)
        if not modifications_map:
            return

        klass_version = SerializableMixIn._get_version_for_class_from_state(state, klass)
        klass_version_current = SerializableMixIn._get_version(klass)
        if klass_version > klass_version_current:
            if _debug:
                logger.debug('got class version {from_state}. Current version {cls}={current}'
                             .format(cls=klass, from_state=klass_version, current=klass_version_current))
            raise OldVersionUnsupported('Tried to restore a model created with a more recent version '
                                        'of PyEMMA. This is not supported! You need at least version {version}'
                                        .format(version=state['pyemma_version']))

        if _debug:
            logger.debug("input state: %s" % state)
        sorted_keys = sorted(modifications_map.keys())
        for key in sorted_keys:
            if not (klass_version_current > key >= klass_version):
                if _debug:
                    logger.debug("skipped interpolation rules for version %s" % key)
                continue
            if _debug:
                logger.debug("processing rules for version %s" % key)
            modifications = modifications_map[key]
            Modifications.apply(modifications, state)
        if _debug:
            logger.debug("interpolated state: %s", state)

    @staticmethod
    def _get_version_for_class_from_state(state, klass):
        """ retrieves the version of the current klass from the state mapping from old locations to new ones. """
        # klass may have renamed, so we have to look this up in the class rename registry.
        names = [_importable_name(klass)]
        # lookup old names, handled by current klass.
        from .util import class_rename_registry
        names.extend(class_rename_registry.old_handled_by(klass))
        for n in names:
            try:
                return state['class_tree_versions'][n]
            except KeyError:
                continue
        # if we did not find a suitable version number return infinity.
        if _debug:
            logger.debug('unable to obtain a _serialize_version for class %s', klass)
        return float('inf')

    def _set_state_from_serializeable_fields_and_state(self, state, klass):
        """ set only fields from state, which are present in klass.__serialize_fields """
        if _debug:
            logger.debug("restoring state for class %s", klass)
        #assert issubclass(klass, SerializableMixIn)
        # handle field renames, deletion, transformations etc.
        SerializableMixIn.__interpolate(state, klass)

        for field in SerializableMixIn._get_serialize_fields(klass):
            if field in state:
                # ensure we can set attributes. Log culprits.
                try:
                    setattr(self, field, state.get(field))
                except AttributeError:
                    logger.exception('field: %s' % field)
            else:
                if _debug:
                    logger.debug("skipped %s, because it is not contained in state", field)

    def __getstate__(self):
        # We just dump the version number for comparison with the actual class.
        # Note: we do not want to set the version number in __setstate__,
        # since we obtain it from the actual definition.
        try:
            if _debug:
                logger.debug('get state of %s' % self)
            #self._version_check()
            state = {'class_tree_versions': {}}
            # currently it is used to handle class renames etc.
            versions = state['class_tree_versions']
            for c in self.__class__.mro():
                name = _importable_name(c)
                try:
                    v = SerializableMixIn._get_version(c)
                # tODO: class version exception should not b e catched?
                except (AttributeError, ClassVersionException):
                    v = -1
                versions[name] = v

            # if we want to save the chain, do this now:
            if self._save_data_producer:
                assert hasattr(self, 'data_producer')
                state['data_producer'] = self.data_producer

            classes_to_inspect = self._get_classes_to_inspect()
            if _debug:
                logger.debug("classes to inspect during setstate: \n%s" % classes_to_inspect)
            for klass in classes_to_inspect:
                self._get_state_of_serializeable_fields(klass, state)

            from pyemma._base.estimator import Estimator
            if isinstance(self, Estimator):
                state.update(Estimator.__my_getstate__(self))

            from pyemma._base.model import Model
            if isinstance(self, Model):
                state.update(Model.__my_getstate__(self))

            from pyemma import version
            state['pyemma_version'] = version

            return state
        except:
            logger.exception('exception during pickling {}'.format(self))
            raise

    def __setstate__(self, state):
        # handle exceptions here, because they will be sucked up by pickle and silently fail...
        try:
            assert state
            # we need to set the model prior extra fields from _serializable_fields, because the model often contains
            # the details needed in the parent estimator.
            if 'model' in state:
                self._model = state.pop('model')

            for klass in self._get_classes_to_inspect():
                self._set_state_from_serializeable_fields_and_state(state, klass=klass)
            state.pop('class_tree_versions')
            state.pop('pyemma_version')

            if hasattr(self, 'data_producer') and 'data_producer' in state:
                self.data_producer = state['data_producer']

            from pyemma._base.estimator import Estimator
            if isinstance(self, Estimator):
                Estimator.__my_setstate__(self, state)

            from pyemma._base.model import Model
            if isinstance(self, Model):
                Model.__my_setstate__(self, state)
        except OldVersionUnsupported as e:
            logger.error(str(e))
            raise
        except:
            logger.exception('exception during unpickling {}'.format(self))
            raise

    def _get_classes_to_inspect(self):
        """ gets classes self derives from which
         1. have custom fields: __serialize_fields
         2. provide a modifications map
         """
        res = list(filter(lambda c:
                          SerializableMixIn._get_version(c, require=False) or
                          (SerializableMixIn._get_serialize_fields(c)
                            or SerializableMixIn._get_interpolation_map(c)),
                          self.__class__.__mro__))
        return res

    def __init_subclass__(cls, *args, **kwargs):
        # ensure, that if this is subclasses, we have a proper class version.
        SerializableMixIn._get_version(cls)
