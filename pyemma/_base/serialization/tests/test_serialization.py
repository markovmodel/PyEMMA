import os
import tempfile
import unittest

import numpy as np

import pyemma
from pyemma._base.serialization.serialization import DeveloperError
from pyemma._base.serialization.serialization import SerializableMixIn, class_rename_registry
from ._test_classes import (test_cls_v1, test_cls_v2, test_cls_v3, _deleted_in_old_version, test_cls_with_old_locations,
                            to_interpolate_with_functions)


class np_container(SerializableMixIn):
    _serialize_version = 0
    _serialize_fields = ('x', 'y', 'z')

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


def patch_old_location(faked_old_class, new_class):
    from pyemma._base.serialization.util import handle_old_classes

    # mark old_loc as being handled by new_class in newer software versions.
    old_loc = "{mod}.{cls}".format(mod=faked_old_class.__module__,
                                   cls=faked_old_class.__name__)
    handle_old_classes(old_loc)(new_class)
    return old_loc


class TestSerialisation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # we do not want these global datastructures to be polluted.
        import copy
        cls.backup_cls_reg = copy.deepcopy(class_rename_registry)
        return cls

    @classmethod
    def tearDownClass(cls):
        class_rename_registry = cls.backup_cls_reg

    def setUp(self):
        self.fn = tempfile.mktemp()
        class_rename_registry.clear()

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

        patch_old_location(test_cls_v1, test_cls_v2)

        inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v2)
        self.assertEqual(inst_restored.z, 42)
        np.testing.assert_equal(inst_restored.b, [1, 2, 3])

    def test_updated_class_v2_to_v3(self):
        inst = test_cls_v2()
        inst.save(self.fn)

        patch_old_location(test_cls_v2, test_cls_v3)

        inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v3)
        self.assertEqual(inst_restored.z, 23)
        self.assertFalse(hasattr(inst_restored, 'y'))

    def test_updated_class_v1_to_v3(self):
        inst = test_cls_v1()
        inst.save(self.fn)

        patch_old_location(test_cls_v1, test_cls_v3)

        inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, test_cls_v3)
        self.assertEqual(inst_restored.z, 23)
        np.testing.assert_equal(inst_restored.c, inst.a)
        self.assertFalse(hasattr(inst_restored, 'y'))

    def test_validate_map_order(self):
        class MapOrder(SerializableMixIn):
            _serialize_version = 0
            _serialize_interpolation_map = {3: [('set', 'x', None)], 0: [('rm', 'x')]}
        s = MapOrder()
        s._validate_interpolation_map(klass=s.__class__)
        self.assertSequenceEqual(list(s._serialize_interpolation_map.keys()),
                                 sorted(s._serialize_interpolation_map.keys()))

    def test_validate_map_invalid_op(self):
        class InvalidOperatorInMap(SerializableMixIn):
            _serialize_version = 0
            _serialize_interpolation_map = {3: [('foo', 'x', None)]}
        s = InvalidOperatorInMap()
        with self.assertRaises(DeveloperError) as cm:
            s._validate_interpolation_map(klass=s.__class__)
        self.assertIn('invalid operation', cm.exception.args[0])

    def test_validate_map_invalid_container_for_actions(self):
        class InvalidContainerInMap(SerializableMixIn):
            _serialize_version = 0
            _serialize_interpolation_map = {3: "foo"}
        s = InvalidContainerInMap()
        with self.assertRaises(DeveloperError) as cm:
            s._validate_interpolation_map(klass=type(s))

        self.assertIn("have to be list or tuple", cm.exception.args[0])

    def test_interpolation_with_map(self):
        c = test_cls_v1()
        c.save(self.fn)
        patch_old_location(test_cls_v1, to_interpolate_with_functions)
        inst_restored = pyemma.load(self.fn)

        self.assertIsInstance(inst_restored, to_interpolate_with_functions)
        self.assertEqual(inst_restored.y, to_interpolate_with_functions.map_y(None))

    def test_renamed_class(self):
        """ ensure a removed class gets properly remapped to an existing one """
        old = _deleted_in_old_version()
        old.save(self.fn)

        # mark old_loc as being handled by test_cls_with_old_locations in newer versions.
        patch_old_location(_deleted_in_old_version, test_cls_with_old_locations)

        # now restore and check it got properly remapped to the new class
        restored = pyemma.load(self.fn)
        # assert isinstance(restored, test_cls_with_old_locations)
        self.assertIsInstance(restored, test_cls_with_old_locations)

    @unittest.skip("not yet impled")
    def test_recent_model_with_old_version(self):
        """ no backward compatibility, eg. recent models are not supported by old version of software. """
        inst = test_cls_v3()
        inst.save(self.fn)
        from pyemma._base.serialization.serialization import OldVersionUnsupported
        test_cls_v3._serialize_version = 0
        if True:
            with self.assertRaises(OldVersionUnsupported) as c:
                pyemma.load(self.fn)
        self.assertIn("need at least {version}".format(version=pyemma.version), c.exception.args[0])

    def test_developer_forgot_to_add_version(self):
        """ we're not allowed to use an un-versioned class """
        with self.assertRaises(DeveloperError):
            class broken(SerializableMixIn): pass
            x = broken()

    def test_evil_things_not_allowed(self):
        """ overwrite the pickling procedure with something an evil method. Ensure it raises."""
        import subprocess
        from pickle import UnpicklingError
        called = False
        def evil(self):
            nonlocal called
            called = True
            return subprocess.Popen, ('/bin/sh', )

        inst = np_container(np.empty(0))
        import types
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

if __name__ == '__main__':
    unittest.main()
