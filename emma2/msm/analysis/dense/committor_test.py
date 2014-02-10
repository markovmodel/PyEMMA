'''
Created on 11.12.2013

@author: jan-hendrikprinz
'''
import unittest
import numpy as np


from emma2.msm.analysis.api import committor

class Test(unittest.TestCase):


    def setUp(self):

        self.T4 = np.array([[0.9, 0.04, 0.03, 0.03], 
                            [0.02, 0.94, 0.02, 0.02], 
                            [0.01, 0.01, 0.94, 0.04], 
                            [0.01, 0.01, 0.08, 0.9]])
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass
    def test_backward_committor(self):
        print committor(self.T4, [0],[3])
        print committor(self.T4, [0],[3] ,forward = False)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()