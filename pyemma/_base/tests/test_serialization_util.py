import unittest

import pyemma._base.serialization.serialization
from pyemma._base.serialization.util import _old_locations


class test_util(unittest.TestCase):
    def test(self):
        """ test class decorator updates the renamed_classes dictionary. """
        old_loc = "pyemma.old.Class"

        @_old_locations([old_loc, ])
        class relocated(object): pass

        assert old_loc in pyemma._base.serialization.serialization._renamed_classes


if __name__ == '__main__':
    unittest.main()
