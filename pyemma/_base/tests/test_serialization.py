import os
import tempfile
import unittest
from io import BytesIO

import numpy as np

import pyemma
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma._base.serialization.util import _old_locations
from pyemma._ext.jsonpickle import dumps, loads


class test_cls_v1(SerializableMixIn):
    _serialize_fields = ('a', 'x', 'y')
    _serialize_version = 1

    def __init__(self):
        self.a = np.random.random((3, 2))
        self.x = 'foo'
        self.y = 0.0

    def __eq__(self, other):
        return np.allclose(self.a, other.a) and self.x == other.x and self.y == other.y


class test_cls_v2(SerializableMixIn):
    _serialize_fields = ('b', 'y', 'z')
    _serialize_version = 2
    # interpolate from version 1: add attr z with value 42
    _serialize_interpolation_map = {1: [('set', 'z', 42),
                                        ('mv', 'a', 'b'),
                                        ('rm', 'x')]}

    def __init__(self):
        self.b = np.random.random((3, 2))
        self.y = 0.0
        self.z = 42

    def __eq__(self, other):
        return np.allclose(self.b, other.b) and self.y == other.y and self.z == other.z


class test_cls_v3(SerializableMixIn):
    # should fake the refactoring of new_cls
    _serialize_fields = ('c', 'z')
    _serialize_version = 3
    # interpolate from version 1 and 2
    _serialize_interpolation_map = {1: [('set', 'z', 42),
                                        ('mv', 'a', 'b'),
                                        ('rm', 'x')],
                                    2: [('set', 'z', 23),
                                        ('mv', 'b', 'c'),
                                        ('rm', 'y')]}

    def __init__(self):
        self.c = np.random.random((3, 2))
        self.z = 23

    def __eq__(self, other):
        return np.allclose(self.c, other.c) and self.y == other.y and self.z == other.z


class _deleted_in_old_version(test_cls_v3):
    _serialize_version = 0
    pass


old_loc = "pyemma._base.tests.test_serialization._deleted_in_old_version"


@_old_locations([old_loc])
class test_cls_with_old_locations(_deleted_in_old_version):
    _serialize_version = 0

    # _serialize_fields = test_cls_v3._serialize_fields

    def __init__(self):
        super(test_cls_with_old_locations, self).__init__()


class TestSerialisation(unittest.TestCase):
    def setUp(self):
        self.fn = tempfile.mktemp()

    def tearDown(self):
        try:
            os.unlink(self.fn)
        except:
            pass

    def test_numpy(self):
        x = np.random.randint(0, 1000, size=1000)
        s = dumps(x)
        actual = loads(s)

        np.testing.assert_equal(actual, x)

    def test_numpy_extracted_dtypes(self):
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

    def test_save_file_like(self):
        from io import BytesIO
        buff = BytesIO()
        t = test_cls_v1()
        t.save(buff)

        buff.seek(0)
        t2 = pyemma.load(buff)
        self.assertEqual(t2, t)

    def test_updated_class(self):
        global test_cls_v1
        old_class = test_cls_v1
        try:
            inst = test_cls_v1()
            inst.save(self.fn)

            test_cls_v1 = test_cls_v2

            inst_restored = pyemma.load(self.fn)

            self.assertIsInstance(inst_restored, test_cls_v2)
            self.assertEqual(inst_restored.z, 42)
        finally:
            test_cls_v1 = old_class

    def test_updated_class_v2_to_v3(self):
        global test_cls_v2
        old_class = test_cls_v2
        try:
            inst = test_cls_v2()
            inst.save(self.fn)

            test_cls_v2 = test_cls_v3

            inst_restored = pyemma.load(self.fn)

            self.assertIsInstance(inst_restored, test_cls_v2)
            self.assertEqual(inst_restored.z, 23)
            self.assertFalse(hasattr(inst_restored, 'y'))
        finally:
            test_cls_v2 = old_class

    def test_updated_class_v1_to_v3(self):
        global test_cls_v1
        old_class = test_cls_v1
        try:
            inst = test_cls_v1()
            inst.save(self.fn)

            test_cls_v1 = test_cls_v3

            inst_restored = pyemma.load(self.fn)

            self.assertIsInstance(inst_restored, test_cls_v3)
            self.assertEqual(inst_restored.z, 23)
            np.testing.assert_equal(inst_restored.c, inst.a)
            self.assertFalse(hasattr(inst_restored, 'y'))
        finally:
            test_cls_v1 = old_class

    def test_validate_map_order(self):
        interpolation_map = {3: [('set', 'x', None)], 0: [('rm', 'x')]}
        s = SerializableMixIn()
        s._serialize_interpolation_map = interpolation_map
        s._validate_interpolation_map()
        self.assertSequenceEqual(list(s._serialize_interpolation_map.keys()),
                                 sorted(s._serialize_interpolation_map.keys()))

    def test_validate_map_invalid_op(self):
        interpolation_map = {3: [('foo', 'x', None)]}
        s = SerializableMixIn()
        s._serialize_interpolation_map = interpolation_map
        from pyemma._base.serialization.serialization import DeveloperError
        with self.assertRaises(DeveloperError) as cm:
            s._validate_interpolation_map()
        self.assertIn('invalid operation', cm.exception.args[0])

    def test_validate_map_invalid_container_for_actions(self):
        interpolation_map = {3: "foo"}
        s = SerializableMixIn()
        s._serialize_interpolation_map = interpolation_map
        from pyemma._base.serialization.serialization import DeveloperError
        with self.assertRaises(DeveloperError) as cm:
            s._validate_interpolation_map()

        self.assertIn("have to be list or tuple", cm.exception.args[0])

    def test_renamed_class(self):
        """ ensure a removed class gets properly remapped to an existing one """
        try:
            buff = BytesIO()

            old = _deleted_in_old_version()
            old.save(buff)
            buff.seek(0)

            # now restore and check it got properly remapped to the new class
            restored = pyemma.load(buff)
            # assert isinstance(restored, test_cls_with_old_locations)
            self.assertIsInstance(restored, test_cls_with_old_locations)
        finally:
            from pyemma._base.serialization.serialization import _renamed_classes
            _renamed_classes.pop(old_loc)


if __name__ == '__main__':
    unittest.main()
