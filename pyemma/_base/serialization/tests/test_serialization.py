import os
import tempfile
import unittest
from io import BytesIO

import numpy as np

import pyemma
from pyemma._base.serialization.serialization import SerializableMixIn, class_rename_registry
from pyemma._ext.jsonpickle import dumps, loads
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

        return np.all(self.x == other.x) and np.all(self.y == other.y)

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
        from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler
        register_ndarray_handler()
        inst = np_container(x)
        inst.save(self.fn)
        restored = inst.load(self.fn)
        self.assertEqual(restored, inst)

    def test_numpy_container_object_array(self):
        x = np.array([None, np.array([1,2,3]), False])
        inst = np_container(x)
        inst.save(self.fn)
        restored = inst.load(self.fn)
        self.assertEqual(restored, inst)

    def test_numpy_extracted_dtypes(self):
        """ scalar values extracted from a numpy array do not posses a python builtin type,
        ensure they are converted to those types properly."""
        value = np.arange(3)
        from pyemma._base.serialization.jsonpickler_handlers import NumpyExtractedDtypeHandler
        for dtype in NumpyExtractedDtypeHandler.np_dtypes:
            converted = value.astype(dtype)[0]
            exported = dumps(converted)
            actual = loads(exported)
            self.assertEqual(actual, converted)

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
        interpolation_map = {3: [('set', 'x', None)], 0: [('rm', 'x')]}
        SerializableMixIn._serialize_interpolation_map = interpolation_map
        s = SerializableMixIn()
        s._serialize_version = 0
        #s._serialize_interpolation_map = interpolation_map
        s._validate_interpolation_map(klass=s.__class__)
        self.assertSequenceEqual(list(s._serialize_interpolation_map.keys()),
                                 sorted(s._serialize_interpolation_map.keys()))

    def test_validate_map_invalid_op(self):
        interpolation_map = {3: [('foo', 'x', None)]}
        SerializableMixIn._serialize_interpolation_map = interpolation_map
        s = SerializableMixIn()
        from pyemma._base.serialization.serialization import DeveloperError
        with self.assertRaises(DeveloperError) as cm:
            s._validate_interpolation_map(klass=s.__class__)
        self.assertIn('invalid operation', cm.exception.args[0])

    def test_validate_map_invalid_container_for_actions(self):
        interpolation_map = {3: "foo"}
        SerializableMixIn._serialize_interpolation_map = interpolation_map
        s = SerializableMixIn()
        from pyemma._base.serialization.serialization import DeveloperError
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
        #with mock.patch(test_cls_v3._serialize_version, '0'):
        test_cls_v3._serialize_version = 0
        if True:
            with self.assertRaises(OldVersionUnsupported) as c:
                pyemma.load(self.fn)
        self.assertIn("need at least {version}".format(version=pyemma.version), c.exception.args[0])


if __name__ == '__main__':
    unittest.main()
