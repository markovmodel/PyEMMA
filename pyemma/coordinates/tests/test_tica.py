'''
Created on 02.02.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.coordinates.io.data_in_memory import DataInMemory
from pyemma.coordinates.transform.tica import TICA
from pyemma.util.log import getLogger

logger = getLogger('TestTICA')


class TestTICA(unittest.TestCase):

    def test(self):
        np.random.seed(0)

        tica = TICA(lag=50, output_dimension=1)
        data = np.random.randn(100, 10)
        ds = DataInMemory(data)
        tica.data_producer = ds

        tica.parametrize()

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

    @unittest.skip("datainmemory does currently not support chunking.")
    def testChunksizeResultsTica(self):
        chunk = 40
        lag = 100
        np.random.seed(0)
        X = np.random.randn(23000, 3)

        # un-chunked
        d = DataInMemory(X)

        tica = TICA(lag=lag, output_dimension=1)
        tica.data_producer = d
        tica.parametrize()

        cov = tica.cov.copy()
        mean = tica.mu.copy()

        # ------- run again with new chunksize -------
        d = DataInMemory(X)
        d.chunksize = chunk
        tica = TICA(lag=lag, output_dimension=1)
        tica.data_producer = d

        tica.parametrize()

        np.testing.assert_allclose(tica.mu, mean)
        np.testing.assert_allclose(tica.cov, cov)

if __name__ == "__main__":
    unittest.main()
