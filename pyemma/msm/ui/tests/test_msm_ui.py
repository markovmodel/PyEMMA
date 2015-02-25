'''
Created on 25.02.2015

@author: marscher
'''
import unittest
import numpy as np

from pyemma.msm.ui.msm import MSM


class TestMSM_UI(unittest.TestCase):

    def testInputList(self):
        dtrajs = [0, 1, 2, 0, 0, 1, 2, 1, 0]
        msm = MSM(dtrajs, 1)

    def testInput1Array(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        msm = MSM(dtrajs, 1)

    def testInputNestedLists(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2]]
        msm = MSM(dtrajs, 1)

    def testInputNestedListsDiffSize(self):
        dtrajs = [[0, 1, 2, 0, 0, 1, 2, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 2, 1]]
        msm = MSM(dtrajs, 1)

    def testInputArray(self):
        dtrajs = np.array([0, 1, 2, 0, 0, 1, 2, 1, 0])
        msm = MSM(dtrajs, 1)

    def testInputArrays(self):
        dtrajs = np.array([[0, 1, 2, 0, 0, 1, 2, 1, 0],
                           [0, 1, 2, 0, 0, 1, 2, 1, 1]])
        msm = MSM(dtrajs, 1)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
