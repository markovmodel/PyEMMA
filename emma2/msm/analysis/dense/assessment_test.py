'''
Created on 07.10.2013

@author: marscher
'''
import unittest

import assessment
import numpy as np


def create_rate_matrix():
    a = [[-3, 3, 0, 0 ],
         [3, -5, 2, 0 ],
         [0, 3, -5, 2],
         [0, 0, 3, -3] ]
    
    return np.asmatrix(a)

class AssessmentDenseTest(unittest.TestCase):

    def setUp(self):
        self.A = create_rate_matrix()

    def tearDown(self):
        pass

    def testIsRateMatrix(self):
        self.assert_(assessment.is_rate_matrix(self.A), \
                     'A should be a rate matrix')
        
        # manipulate matrix so it isn't a rate matrix any more
        self.A[0][0] = 3
        self.assertFalse(assessment.is_rate_matrix(self.A), \
                        'matrix is not a rate matrix')

if __name__ == "__main__":
    unittest.main()
