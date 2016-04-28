import tempfile
import unittest

import numpy as np
from jsonpickle import dumps, loads

from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler, unregister_ndarray_handler
from pyemma._base.serialization.serialization import SerializableMixIn


class TestSerialisation(unittest.TestCase):
    def test_numpy(self):
        try:
            register_ndarray_handler()

            x = np.random.randint(0, 1000, size=100000)
            s = dumps(x)
            actual = loads(s)

            np.testing.assert_equal(actual, x)
        finally:
            unregister_ndarray_handler()

    def test_save_interface(self):
        class test_cls(SerializableMixIn):
            def __init__(self):
                self.a = np.random.random((100, 10))
                self.x = 'foo'
                self.y = None

        inst = test_cls()
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            inst.save(filename=fh.name)

            new = test_cls.load(fh.name)

            self.assertEqual(new, inst)
