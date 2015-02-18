import unittest
import numpy as np

from pyemma.coordinates.acf import acf

class TestTICA(unittest.TestCase):
    def test(self):
        # generate some data
        data = np.random.rand(100,3)

        testacf = acf(data)

        # direct computation of acf (single trajectory, three observables)
        N = data.shape[0]
        refacf = np.zeros(data.shape)
        meanfree = data - np.mean(data,axis=0)
        padded = np.concatenate((meanfree,np.zeros(data.shape)),axis=0)
        for tau in xrange(N):
            refacf[tau] = (padded[0:N,:]*padded[tau:N+tau,:]).sum(axis=0)/(N-tau)
        refacf /= refacf[0] # normalize

        np.testing.assert_allclose(refacf,testacf)
        
if __name__ == "__main__":
    unittest.main()
