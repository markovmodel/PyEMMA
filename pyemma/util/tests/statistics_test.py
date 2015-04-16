
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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