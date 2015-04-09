'''
Created on 09.04.2015

@author: marscher
'''
import unittest

import numpy as np
from pyemma.coordinates.io.data_in_memory import DataInMemory
from pyemma.coordinates.clustering.uniform_time import UniformTimeClustering


class TestUniformTimeClustering(unittest.TestCase):

    def test_1d(self):
        x = np.random.random(1000)
        reader = DataInMemory(x)

        k = 2
        c = UniformTimeClustering(k)

        c.data_producer = reader
        c.parametrize()

    def test_2d(self):
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 2
        c = UniformTimeClustering(k)

        c.data_producer = reader
        c.parametrize()

    def test_big_k(self):
        # TODO: fix this (some error handling should be done in _param_init)
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 298
        c = UniformTimeClustering(k)

        c.data_producer = reader
        c.parametrize()


if __name__ == "__main__":
    unittest.main()
