'''
Created on 18.10.2013

@author: marscher
'''
import unittest

import emma2.msm.analysis.api as api
import numpy as np

class Test(unittest.TestCase):

    def testTPT(self):
        A = np.ndarray([1, 2, 3], dtype=int)
        B = np.ndarray([4, 2], dtype=int)
        
        T = np.ndarray([[ 0.5, 0, 0.5, 0],
           [0, 0.5, 0.5, 0],
           [1 / 3., 1 / 3., 0, 1 / 3.],
           [0, 0, 1, 0]], shape=(4,4), dtype=np.double)

        itpt = api.tpt(T, A, B)
        
        print "flux: ", itpt.getFlux()
        print "net flux: ", itpt.getNetFlux()
        print "total flux: ", itpt.getTotalFlux()
        print "forward committor", itpt.getForwardCommittor()
        print "backward committor", itpt.getBackwardCommitor()
        
if __name__ == "__main__":
    unittest.main()
