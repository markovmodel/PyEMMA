
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import six

import pyemma
from pyemma._base.serialization.serialization import ClassVersionException
from pyemma._base.serialization.serialization import SerializableMixIn
from ._test_classes import (test_cls_v1, test_cls_v2, test_cls_v3, _deleted_in_old_version, test_cls_with_old_locations,
                            to_interpolate_with_functions, )


class np_container(SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = ('x', 'y', 'z')

    def __init__(self, x):
        self.x = x
        self.y = x
        self.z = [x, x]

    def __eq__(self, other):
        if not isinstance(other, np_container):
            return False

        for e0, e1 in zip(self.x, other.x):
            np.testing.assert_equal(e0, e1)

        return True

class private_attr(SerializableMixIn):
    __serialize_fields = ('_private_attr___foo',)
    __serialize_version = 0

    def __init__(self):
        self.___foo = None

    def has_private_attr(self):
        try:
            assert self.___foo is None
            return True
        except AttributeError:
            return False


@contextmanager
def patch_old_location(faked_old_class, new_class):
    # register new_class for current context as replacement for faked_old_class
    from pyemma._base.serialization.util import handle_old_classes, class_rename_registry
    import copy
    from unittest import mock
    my_copy = copy.deepcopy(class_rename_registry)
    # mark old_loc as being handled by new_class in newer software versions.
    old_loc = "{mod}.{cls}".format(mod=faked_old_class.__module__,
                                   cls=faked_old_class.__name__)
    with mock.patch('pyemma._base.serialization.util.class_rename_registry', my_copy):
        handle_old_classes(old_loc)(new_class)
        from pyemma._base.serialization.util import class_rename_registry as c
        assert c.find_replacement_for_old(old_loc) == new_class

        yield

@unittest.skipIf(six.PY2, 'only py3')
class TestSerialisation(unittest.TestCase):

    def setUp(self):
        self.fn = tempfile.mktemp()

    def tearDown(self):
        try:
            os.unlink(self.fn)
        except:
            pass

    def test_numpy_container(self):
        x = np.random.randint(0, 1000, size=10)
        inst = np_container(x)
        inst.save(self.fn)
        restored = inst.load(self.fn)
        self.assertEqual(restored, inst)

    def test_numpy_container_object_array(self):
        x = np.array([None, np.array([1, 2, 3]), False])
        inst = np_container(x)
        inst.save(self.fn)
        restored = inst.load(self.fn)
        self.assertEqual(restored, inst)

    def test_save_interface(self):
        inst = test_cls_v1()
        inst.save(self.fn)
        new = test_cls_v1.load(self.fn)
        self.assertEqual(new, inst)

    def test_updated_class_v1_to_v2(self):
        """ """
        inst = test_cls_v1()
        inst.save(self.fn)

        with patch_old_location(test_cls_v1, test_cls_v2):
            inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v2)
        self.assertEqual(inst_restored.z, 42)
        np.testing.assert_equal(inst_restored.b, inst.a)

    def test_updated_class_v2_to_v3(self):
        inst = test_cls_v2()
        inst.save(self.fn)

        with patch_old_location(test_cls_v2, test_cls_v3):
            inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v3)
        self.assertEqual(inst_restored.z, 23)
        self.assertFalse(hasattr(inst_restored, 'y'))

    def test_updated_class_v1_to_v3(self):
        inst = test_cls_v1()
        inst.save(self.fn)

        with patch_old_location(test_cls_v1, test_cls_v3):
            inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v3)
        self.assertEqual(inst_restored.z, 23)
        np.testing.assert_equal(inst_restored.c, inst.a)
        self.assertFalse(hasattr(inst_restored, 'y'))

    def test_interpolation_with_map(self):
        c = test_cls_v1()
        c.save(self.fn)
        with patch_old_location(test_cls_v1, to_interpolate_with_functions):
            inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, to_interpolate_with_functions)
        self.assertEqual(inst_restored.y, to_interpolate_with_functions.map_y(None))

    def test_renamed_class(self):
        """ ensure a removed class gets properly remapped to an existing one """
        old = _deleted_in_old_version()
        old.save(self.fn)

        # mark old_loc as being handled by test_cls_with_old_locations in newer versions.
        with patch_old_location(_deleted_in_old_version, test_cls_with_old_locations):
            # now restore and check it got properly remapped to the new class
            restored = pyemma.load(self.fn)
        # assert isinstance(restored, test_cls_with_old_locations)
        self.assertIsInstance(restored, test_cls_with_old_locations)

    def test_recent_model_with_old_version(self):
        """ no backward compatibility, eg. recent models are not supported by old version of software. """
        inst = test_cls_v3()
        inst.save(self.fn)
        from pyemma._base.serialization.serialization import OldVersionUnsupported
        old = SerializableMixIn._get_version(inst.__class__)
        def _set_version(cls, val):
            setattr(cls, '_%s__serialize_version' % cls.__name__, val)
        _set_version(test_cls_v3, 0)
        try:
            with self.assertRaises(OldVersionUnsupported) as c:
                pyemma.load(self.fn)
            self.assertIn("need at least version {version}".format(version=pyemma.version), c.exception.args[0])
        finally:
            _set_version(test_cls_v3, old)

    def test_developer_forgot_to_add_version(self):
        """ we're not allowed to use an un-versioned class """
        with self.assertRaises(ClassVersionException):
            class broken(SerializableMixIn): pass
            x = broken()

    def test_evil_things_not_allowed(self):
        """ overwrite the pickling procedure with something an evil method. Ensure it raises."""
        import subprocess
        from pickle import UnpicklingError
        import types
        called = {'result': False}
        def evil(self):
            called['result'] = True
            return subprocess.Popen, ('/bin/sh', )

        inst = np_container(np.empty(0))
        old = SerializableMixIn.__getstate__
        old2 = inst.__class__.__reduce__
        try:
            del SerializableMixIn.__getstate__
            inst.__class__.__reduce__ = types.MethodType(evil, inst)
            inst.save(self.fn)
            with self.assertRaises(UnpicklingError) as e:
                pyemma.load(self.fn)
            self.assertIn('not allowed', str(e.exception))
            self.assertTrue(called, 'hack not executed')
        finally:
            SerializableMixIn.__getstate__ = old
            np_container.__reduce__ = old2

    def test_private(self):
        inst = private_attr()
        inst.save(self.fn)
        restore = pyemma.load(self.fn)
        assert restore.has_private_attr()

    def test_rename(self):
        inst = np_container(None)
        inst.save(self.fn, model_name='a')
        from pyemma._base.serialization.h5file import H5File
        with H5File(self.fn) as f:
            f.rename('a', 'b')
            models = f.models_descriptive.keys()
        self.assertIn('b', models)
        self.assertNotIn('a', models)

    def test_delete(self):
        inst = np_container(None)
        inst.save(self.fn, model_name='a')
        from pyemma._base.serialization.h5file import H5File
        with H5File(self.fn) as f:
            f.delete('a')
            self.assertNotIn('a', f.models_descriptive.keys())


if __name__ == '__main__':
    unittest.main()
