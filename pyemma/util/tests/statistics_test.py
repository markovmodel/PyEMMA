'''
Created on Jul 25, 2014

@author: noe
'''
import unittest
from pyemma.util import statistics
import numpy as np


class TestStatistics(unittest.TestCase):

    def assertConfidence(self, sample, alpha, precision):
        alpha = 0.5
        conf = statistics.confidence_interval(sample, alpha)

        n_in = 0.0
        for i in range(len(sample)):
            if sample[i] > conf[1] and sample[i] < conf[2]:
                n_in += 1.0

        assert(alpha - (n_in/len(sample)) < precision)

    def test_confidence_interval(self):
        # exponential distribution
        self.assertConfidence(np.random.exponential(size=10000), 0.5, 0.01)
        self.assertConfidence(np.random.exponential(size=10000), 0.8, 0.01)
        self.assertConfidence(np.random.exponential(size=10000), 0.95, 0.01)
        # Gaussian distribution
        self.assertConfidence(np.random.normal(size=10000), 0.5, 0.01)
        self.assertConfidence(np.random.normal(size=10000), 0.8, 0.01)
        self.assertConfidence(np.random.normal(size=10000), 0.95, 0.01)

if __name__ == "__main__":
    unittest.main()