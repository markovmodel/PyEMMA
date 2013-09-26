'''
Created on 25.09.2013

@author: marscher
'''
import unittest
from emma2.test.test_msm import _TestMSM

class Emma2TestSuite(unittest.TestSuite):
    '''
    Main test suite for emma2.
    Add classes of instance unittest.suite.testsuite or unittest.TestCase to 
    constructor.
    '''
    def __init__(self, params):
        '''
        Constructor
        '''
        self.addTest(_TestMSM)
        
if __name__ == '__main__':
    print "running test suite with xml runner"
    unittest.main()