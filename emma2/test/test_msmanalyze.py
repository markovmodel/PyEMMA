'''
Created on 24.09.2013

@author: marscher
'''
import unittest

from emma2.msm.analyze import *

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testCommitorForward(self):
        committor_forward(T, A, B)
        self.assertAlmostEqual(first, second, places, msg, delta)
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()