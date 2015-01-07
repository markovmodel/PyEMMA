r"""Unit tests for the mean first passage time module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from mean_first_passage_time import mfpt

class TestMfpt(unittest.TestCase):
    
    def setUp(self):
        p1=0.7
        p2=0.5
        q2=0.4
        q3=0.6
        """3x3 birth and death transition matrix"""
        self.P=np.array([[1.0-p1, p1, 0.0],\
                             [q2, 1.0-q2-p2, p2],\
                             [0, q3, 1.0-q3]]) 
        """Vector of mean first passage times to target state t=0"""
        self.m0=np.array([0.0, (p2+q3)/(q2*q3), (p2+q2+q3)/(q2*q3)])
        """Vector of mean first passage times to target state t=1"""
        self.m1=np.array([1.0/p1, 0.0, 1.0/q3])
        """Vector of mean first passage times to target state t=2"""
        self.m2=np.array([(p2+p1+q2)/(p1*p2), (p1+q2)/(p1*p2), 0.0])

    def tearDown(self):
        pass

    def test_mfpt(self):
        x=mfpt(self.P, 0)
        assert_allclose(x, self.m0)

        x=mfpt(self.P, 1)
        assert_allclose(x, self.m1)

        x=mfpt(self.P, 2)
        assert_allclose(x, self.m2)
        
if __name__=="__main__":
    unittest.main()
