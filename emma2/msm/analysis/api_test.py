'''
Created on 18.10.2013

@author: marscher
'''
import unittest

import emma2.msm.analysis.api as api
import numpy as np

class Test(unittest.TestCase):

    def testTPT(self):
        A = np.asarray([0], dtype=int)
        B = np.asarray([5], dtype=int)
        C = np.ndarray(buffer=np.array(
            [[6000, 3, 0, 0, 0, 0],
             [3, 1000, 3, 0, 0, 0],
             [0, 3, 1000, 3, 0, 0],
             [0, 0, 3, 1000, 3, 0],
             [0, 0, 0, 3, 1000, 3],
             [0, 0, 0, 0, 3, 90000]]), shape=(6, 6))
        
        T = C / np.sum(C, axis=1)[:, np.newaxis]

        itpt = api.tpt(T, A, B)
        
        print "flux: ", itpt.getFlux()
        print "net flux: ", itpt.getNetFlux()
        print "total flux: ", itpt.getTotalFlux()
        print "forward committor", itpt.getForwardCommittor()
        print "backward committor", itpt.getBackwardCommittor()
        
if __name__ == "__main__":
    unittest.main()
