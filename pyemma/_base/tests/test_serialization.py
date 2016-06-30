import os
import tempfile
import unittest

import numpy as np

import pyemma
from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler, unregister_ndarray_npz_handler
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma._ext.jsonpickle import dumps, loads


class test_cls(SerializableMixIn):
    _serialize_fields = ('a', 'x', 'y')
    _serialize_version = 1

    def __init__(self):
        self.a = np.random.random((3, 2))
        self.x = 'foo'
        self.y = 0.0

    def __getstate__(self):
        state = super(test_cls, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=test_cls))
        return state

    def __setstate__(self, state):
        self._set_state_from_serializeable_fields_and_state(state, test_cls)

    def __eq__(self, other):
        return np.allclose(self.a, other.a) and self.x == other.x and self.y == other.y


class new_cls(SerializableMixIn):
    _serialize_fields = ('b', 'y', 'z')
    _serialize_version = 2
    # interpolate from version 1: add attr z with value 42
    _serialize_interpolation_map = {1 : [('set', 'z', 42),
                                         ('mv', 'a', 'b'),
                                         ('rm', 'x')]}

    def __init__(self):
        self.b = np.random.random((3, 2))
        self.y = 0.0
        self.z = 42

    def __getstate__(self):
        state = super(test_cls, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=new_cls))
        return state

    def __setstate__(self, state):
        self._set_state_from_serializeable_fields_and_state(state, new_cls)

    def __eq__(self, other):
        return np.allclose(self.b, other.b) and self.y == other.y and self.z == other.z


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
        inst = test_cls()
        try:
            with tempfile.NamedTemporaryFile(delete=False) as fh:
                inst.save(filename=fh.name)
                new = test_cls.load(fh.name)
                self.assertEqual(new, inst)
        finally:
            os.unlink(fh.name)

    def test_updated_class(self):
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            inst = test_cls()
            inst.save(f.name)

            global test_cls
            test_cls = new_cls

            inst_restored = pyemma.load(f.name)

            self.assertIsInstance(inst_restored, new_cls)
            self.assertEqual(inst_restored.z, 42)
        finally:
            os.unlink(f.name)