import unittest
from unittest import TestCase
import numpy as np
import pyemma.coordinates as coor


class TestRandomAccessStride(TestCase):
    def setUp(self):
        self.dim = 5
        self.data = [np.random.random((100, self.dim)),
                     np.random.random((1, self.dim)),
                     np.random.random((2, self.dim))]
        self.stride = {0: [1, 2, 3], 2: [1]}

    def test_data_in_memory_random_access(self):
        # access with a chunk_size that is larger than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=10)
        out1 = data_in_memory.get_output(stride=self.stride)

        # access with a chunk_size that is smaller than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=1)
        out2 = data_in_memory.get_output(stride=self.stride)

        # access in full trajectory mode
        data_in_memory = coor.source(self.data, chunk_size=0)
        out3 = data_in_memory.get_output(stride=self.stride)

        for idx in self.stride.keys():
            np.testing.assert_array_equal(out1[idx], out2[idx])
            np.testing.assert_array_equal(out2[idx], out3[idx])

    def test_data_in_memory_random_access_with_lag(self):
        # do we need this?
        pass


if __name__ == '__main__':
    unittest.main()
