import unittest
import numpy as np
from jsonpickle import dumps, loads
from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler, unregister_ndarray_handler


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
