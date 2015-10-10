'''
Created on 23.07.2015

@author: marscher
'''
import unittest
import numpy as np
from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates.data.sparsifier import Sparsifier
from pyemma.coordinates.api import tica


class TestSparsifier(unittest.TestCase):

    def setUp(self):
        self.X = np.random.random((1000, 10))
        ones = np.ones((1000, 1))
        data = np.concatenate((self.X, ones), axis=1)
        self.src = DataInMemory(data)
        self.src.chunksize = 200

        self.sparsifier = Sparsifier()
        self.sparsifier.data_producer = self.src
        self.sparsifier.parametrize()

    def test_constant_column(self):
        out = self.sparsifier.get_output()[0]
        np.testing.assert_allclose(out, self.X)

    def test_constant_column_tica(self):
        tica_obj = tica(self.sparsifier, kinetic_map=True, var_cutoff=1)
        self.assertEqual(tica_obj.dimension(), self.sparsifier.dimension())

    def test_numerical_close_features(self):
        rtol = self.sparsifier.rtol
        noise = (rtol*200) * (np.random.random(1000)*2 - 0.5)
        self.src._data[0][:, -1] += noise

        out = self.sparsifier.get_output()[0]
        np.testing.assert_allclose(out, self.X)


if __name__ == "__main__":
    unittest.main()
