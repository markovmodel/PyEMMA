'''
Created on 24.09.2013

@author: marscher
'''
import unittest
import numpy as np
from emma2.msm.util import *

class Test(unittest.TestCase):

    def testIsTransitionMatrix(self):
        """
        tests the function msm.util.isTransitionMatrix(T)
        """
        size = 10
        # create count matrix with samples between [0, 100] and shape size^2
        # and cast it to float
        C = np.asarray(np.random.randint(0, 100, size=(size, size)), np.float, 2)
        row_sums = C.sum(axis = 1)
        for i in xrange(size):
            C[i] /= row_sums[i]
        
        self.assert_(util.isTransitionMatrix(C), "T should be a transition matrix")

if __name__ == "__main__":
    unittest.main()