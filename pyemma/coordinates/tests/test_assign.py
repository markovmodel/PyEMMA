'''
Created on 13.04.2015

@author: marscher
'''
import unittest
import numpy as np
from pyemma.coordinates.api import assign_to_centers


class TestAssignCenters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.random.random((100, 3))
        cls.centers = np.random.random((5, 3))

    def test_assign(self):
        assign = assign_to_centers(self.data, self.centers, stride=1)

    def test_assign_stride(self):
        assign = assign_to_centers(self.data, self.centers, stride=2)

        self.assertEqual(assign[0].shape[0], self.data[::2].shape[0])

if __name__ == "__main__":
    unittest.main()
