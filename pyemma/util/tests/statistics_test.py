
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 25, 2014

@author: noe
'''

from __future__ import absolute_import
import unittest
from pyemma.util import statistics
import numpy as np
from six.moves import range


class TestStatistics(unittest.TestCase):

    def assertConfidence(self, sample, alpha, precision):
        alpha = 0.5
        conf = statistics.confidence_interval(sample, alpha)

        n_in = 0.0
        for i in range(len(sample)):
            if sample[i] > conf[0] and sample[i] < conf[1]:
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