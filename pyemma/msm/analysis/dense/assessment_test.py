
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
Created on 07.10.2013

@author: marscher
'''
import unittest
import numpy as np

import assessment
from committor_test import BirthDeathChain


def create_rate_matrix():
    a = [[-3, 3, 0, 0],
         [3, -5, 2, 0],
         [0, 3, -5, 2],
         [0, 0, 3, -3]]
    
    return np.asmatrix(a)


class RateMatrixTest(unittest.TestCase):

    def setUp(self):
        self.A = create_rate_matrix()

    def testIsRateMatrix(self):
        self.assert_(assessment.is_rate_matrix(self.A), 'A should be a rate matrix')
        
        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(assessment.is_rate_matrix(self.A), 'matrix is not a rate matrix')
        

class ReversibleTest(unittest.TestCase):
    
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()
        self.mu = self.bdc.stationary_distribution()
    
    def testIsReversible(self):
        # create a reversible matrix
        self.assertTrue(assessment.is_reversible(self.T, self.mu), "T should be reversible")

if __name__ == "__main__":
    unittest.main()