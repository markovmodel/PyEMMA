import unittest
from pyemma.util.units import bytes_to_string

class TestUnits(unittest.TestCase):

    def test_human_readable_byte_size(self):
        n_bytes = 1393500160  # about 1.394 GB
        self.assertEqual(bytes_to_string(n_bytes), "1.3GB", "this number of bytes should "
                                                            "result in 1.3GB and not in %s" % bytes_to_string(n_bytes))
        self.assertEqual(bytes_to_string(0), "0B", "0 bytes should result in \"0B\"")