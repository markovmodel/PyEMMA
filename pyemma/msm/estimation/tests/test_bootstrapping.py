'''
Created on Jul 25, 2014

@author: noe
'''
import unittest
import numpy as np
import pyemma.msm.estimation as msmest

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def validate_counts(self, ntraj, length, n, tau):
        dtrajs = []
        for i in range(ntraj):
            dtrajs.append(np.random.random_integers(0, n-1, size = length))
        for i in range(10):
            C = msmest.bootstrap_counts(dtrajs, tau).toarray()
            assert(np.shape(C) == (n,n))
            assert(np.sum(C) == (ntraj*length) / tau)

    def test_bootstrap_counts(self):
        self.validate_counts(1, 10000, 10, 10)
        self.validate_counts(1, 10000, 100, 1000)
        self.validate_counts(10, 100, 2, 10)
        self.validate_counts(10, 1000, 100, 100)
        self.validate_counts(1000, 10, 1000, 1)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()