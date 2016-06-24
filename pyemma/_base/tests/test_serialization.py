import tempfile
import unittest

import numpy as np
from pyemma._ext.jsonpickle import dumps, loads

from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler, unregister_ndarray_npz_handler
from pyemma._base.serialization.serialization import SerializableMixIn


class test_cls(SerializableMixIn):
    _serialize_fields = ('a', 'x', 'y')
    _version = 1

    def __init__(self):
        self.a = np.random.random((3, 2))
        self.x = 'foo'
        self.y = None

    def __getstate__(self):
        state = super(test_cls, self).__getstate__()
        state.update(self._get_state_of_serializeable_fields(klass=test_cls))
        return state

    def __eq__(self, other):
        return np.allclose(self.a, other.a) and self.x == other.x and self.y == other.y


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
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            inst.save(filename=fh.name)

            new = test_cls.load(fh.name)

            self.assertEqual(new, inst)
