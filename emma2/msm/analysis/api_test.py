'''
Created on 18.10.2013

@author: marscher
'''
import unittest

import emma2.msm.analysis.api as api
import numpy as np

import logging
log = logging.getLogger(__name__)

class Test(unittest.TestCase):
    
    def setUp(self):
        from emma2.util.pystallone import stallone_available
        if not stallone_available:
            self.skipTest('stallone not installed')
        C = np.ndarray(buffer=np.array(
            [[6000, 3, 0, 0, 0, 0],
             [3, 1000, 3, 0, 0, 0],
             [0, 3, 1000, 3, 0, 0],
             [0, 0, 3, 1000, 3, 0],
             [0, 0, 0, 3, 1000, 3],
             [0, 0, 0, 0, 3, 90000]]), shape=(6, 6))
        #size = 50
        #C = np.random.random_integers(1, 100, size=(size,size))
        self.T = 1.0 * C / np.sum(C, axis=1)[:, np.newaxis]

    def testTPT(self):
        A = np.asarray([0])
        B = np.asarray([len(self.T)-1])

        itpt = api.tpt(self.T, A, B)
        
        log.info("flux:\n%s" % itpt.getFlux())
        log.info("net flux:\n%s" % itpt.getNetFlux())
        log.info("total flux:\n%s" % itpt.getTotalFlux())
        log.info("forward committor:\n%s" % itpt.getForwardCommittor())
        log.info("backward committor:\n%s" % itpt.getBackwardCommittor())
        
if __name__ == "__main__":
    unittest.main()
