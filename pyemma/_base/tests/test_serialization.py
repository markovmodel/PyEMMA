import os
import tempfile
import unittest

import numpy as np

import pyemma
from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler, unregister_ndarray_npz_handler
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma._ext.jsonpickle import dumps, loads


class test_cls_v1(SerializableMixIn):
    _serialize_fields = ('a', 'x', 'y')
    _serialize_version = 1

    def __init__(self):
        self.a = np.random.random((3, 2))
        self.x = 'foo'
        self.y = 0.0

    def __getstate__(self):
        state = super(test_cls_v1, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=test_cls_v1))
        return state

    def __setstate__(self, state):
        self._set_state_from_serializeable_fields_and_state(state, test_cls_v1)

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

    def __getstate__(self):
        state = super(test_cls_v2, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=test_cls_v2))
        return state

    def __setstate__(self, state):
        self._set_state_from_serializeable_fields_and_state(state, test_cls_v2)

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

    def __getstate__(self):
        state = super(test_cls_v3, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=test_cls_v3))
        return state

    def __setstate__(self, state):
        self._set_state_from_serializeable_fields_and_state(state, test_cls_v3)

    def __eq__(self, other):
        return np.allclose(self.c, other.c) and self.y == other.y and self.z == other.z


class TestSerialisation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_ndarray_handler()

    def test_numpy(self):
        try:
            x = np.random.randint(0, 1000, size=100000)
            s = dumps(x)
            actual = loads(s)

            np.testing.assert_equal(actual, x)
        finally:
            unregister_ndarray_npz_handler()

    def test_save_interface(self):
        inst = test_cls_v1()
        try:
            with tempfile.NamedTemporaryFile(delete=False) as fh:
                inst.save(filename=fh.name)
                new = test_cls_v1.load(fh.name)
                self.assertEqual(new, inst)
        finally:
            os.unlink(fh.name)

    def test_updated_class(self):
        old_class = test_cls_v1
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            inst = test_cls_v1()
            inst.save(f.name)

            global test_cls_v1
            test_cls_v1 = test_cls_v2

            inst_restored = pyemma.load(f.name)

            self.assertIsInstance(inst_restored, test_cls_v2)
            self.assertEqual(inst_restored.z, 42)
        finally:
            os.unlink(f.name)
            test_cls_v1 = old_class

    def test_updated_class_v2_to_v3(self):
        old_class = test_cls_v2
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            inst = test_cls_v2()
            inst.save(f.name)

            global test_cls_v2
            test_cls_v2 = test_cls_v3

            inst_restored = pyemma.load(f.name)

            self.assertIsInstance(inst_restored, test_cls_v2)
            self.assertEqual(inst_restored.z, 23)
            self.assertFalse(hasattr(inst_restored, 'y'))
        finally:
            os.unlink(f.name)
            test_cls_v2 = old_class

    def test_updated_class_v1_to_v3(self):
        old_class = test_cls_v1
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            inst = test_cls_v1()
            inst.save(f.name)

            global test_cls_v1
            test_cls_v1 = test_cls_v3

            inst_restored = pyemma.load(f.name)

            self.assertIsInstance(inst_restored, test_cls_v3)
            self.assertEqual(inst_restored.z, 23)
            np.testing.assert_equal(inst_restored.c, inst.a)
            self.assertFalse(hasattr(inst_restored, 'y'))
        finally:
            os.unlink(f.name)
            test_cls_v1 = old_class

    def test_validate_map_order(self):
        interpolation_map = {3: [('set', 'x', None)], 0: [('rm', 'x')]}
        s = SerializableMixIn()
        s._serialize_interpolation_map = interpolation_map
        s._validate_interpolation_map()
        is_sorted = lambda l: all(l[i] <= l[i+1] for i in xrange(len(l)-1))
        self.assertTrue(is_sorted(s._serialize_interpolation_map.keys()))

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

        self.assertIn("contain", cm.exception.args[0])
