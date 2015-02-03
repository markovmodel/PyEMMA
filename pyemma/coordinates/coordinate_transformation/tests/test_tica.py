'''
Created on 02.02.2015

@author: marscher
'''
import unittest

from pyemma.coordinates.coordinate_transformation.io.data_in_memory import DataInMemory
from pyemma.coordinates.coordinate_transformation.transform.tica import TICA
import numpy as np


class TestTICA(unittest.TestCase):

    def testName(self):
        np.random.seed(0)

        tica = TICA(lag=50, output_dimension=1)
        data = np.random.randn(100, 10)
        ds = DataInMemory(data)
        tica.data_producer = ds

        tica.parametrize()
        # print tica.cov_tau
        # print tica.eigenvalues
        # print tica.eigenvectors

        Y = tica.map(data)

    def test_duplicated_data(self):
        tica = TICA(lag=1, output_dimension=1)

        # make some data that has one column repeated twice
        X = np.random.randn(100, 2)
        X = np.hstack((X, X[:, 0, np.newaxis]))

        d = DataInMemory(X)

        tica.data_producer = d
        tica.parametrize()

        assert tica.eigenvectors.dtype == np.float64
        assert tica.eigenvalues.dtype == np.float64

    def test_singular_zeros(self):
        tica = TICA(lag=1, output_dimension=1)

        # make some data that has one column of all zeros
        X = np.random.randn(100, 2)
        X = np.hstack((X, np.zeros((100, 1))))

        d = DataInMemory(X)

        tica.data_producer = d
        tica.parametrize()

        assert tica.eigenvectors.dtype == np.float64
        assert tica.eigenvalues.dtype == np.float64

if __name__ == "__main__":
    unittest.main()
