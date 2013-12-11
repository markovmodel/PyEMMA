'''
Created on 06.12.2013

@author: Jan-Hendrik Prinz

This module provides unit tests for the sensitivity module

Most tests consist of the comparison of some (randomly selected) sensitivity matrices 
against numerical differentiation results.
'''

import unittest
import numpy as np

import sensitivity

from emma2.msm.analysis.api import eigenvalue_sensitivity, mfpt_sensitivity, committor_sensitivity, eigenvector_sensitivity

class TestExpectations(unittest.TestCase):
    def setUp(self):
        
        self.T=np.array([[0.8, 0.2], [0.05, 0.95]])
        
        self.S0=np.array([[0.2, 0.2], [0.8, 0.8]])
        self.S1=np.array([[0.8, -0.2], [-0.8, 0.2]])
        
        self.T4 = np.array([[0.9, 0.04, 0.03, 0.03], 
                            [0.02, 0.94, 0.02, 0.02], 
                            [0.01, 0.01, 0.94, 0.04], 
                            [0.01, 0.01, 0.08, 0.9]])
        
        self.qS42 = np.array([[0., 0., 0., 0.], 
                              [0., 1.7301, 2.24913, 2.94118], 
                              [0., 10.3806, 13.4948, 17.6471], 
                              [0., 0., 0., 0.]]);
                              
        self.qS41 = np.array([[0., 0., 0., 0.], 
                              [0., 10.3806, 13.4948, 17.6471], 
                              [0., 3.46021, 4.49826, 5.88235], 
                              [0., 0., 0., 0.]])
        
        self.S4zero = np.zeros((4,4))
        
        self.mS01 = np.array(
                              [[0., 0., 0., 0.], 
                               [0., 1875., 2187.5, 2187.5], 
                               [0., 2410.71, 2812.5, 2812.5], 
                               [0., 1339.29, 1562.5, 1562.5]]
                              )
        
        self.mS02 = np.array(
                             [[0., 0., 0., 0.], 
                              [0., 937.5, 1093.75, 1093.75], 
                              [0., 3883.93, 4531.25, 4531.25], 
                              [0., 1741.07, 2031.25, 2031.25]]
                             )
        
        self.mS32 = np.array(
                             [[102.959, 114.793, 87.574, 0.], 
                              [180.178, 200.888, 153.254, 0.],
                              [669.231, 746.154, 569.231, 0.], 
                              [0., 0., 0., 0.]]
                            )
                
        self.mV11 = np.array(
                             [[-3.4819290, -6.6712389, 2.3317857, 2.3317857],
                              [1.4582191, 2.7938918, -0.9765401, -0.9765414],
                              [-0.7824563, -1.4991658, 0.5239938, 0.5239950],
                              [-0.2449557, -0.4693191, 0.1640369, 0.1640476]]
                             )

        self.mV22 = np.array(
                             [[0.0796750, -0.0241440, -0.0057555, -0.0057555],
                              [-2.2829491, 0.6918640, 0.1649531, 0.1649531],
                              [-5.8183459, 1.7632923, 0.4203993, 0.4203985],
                              [16.4965144, -4.9993827, -1.1919380, -1.1919347]]
                             )
        
        self.mV03 = np.array(
                             [[1.3513524, 1.3513531, 1.3513533, 1.3513533],
                              [2.3648662, 2.3648656, 2.3648655, 2.3648656],
                              [-0.6032816, -0.6032783, -0.6032800, -0.6032799],
                              [-3.1129331, -3.1129331, -3.1129321, -3.1129312]]
                             )

                
        pass
    def tearDown(self):
        pass

    def test_eigenvalue_sensitivity(self):
                
        self.assertTrue(np.allclose(eigenvalue_sensitivity(self.T,0), self.S0))      
        self.assertTrue(np.allclose(eigenvalue_sensitivity(self.T,1), self.S1))      
        
    def test_forward_committor_sensitivity(self):
            
        self.assertTrue(np.allclose(committor_sensitivity(self.T4, [0], [3], 0), self.S4zero))      
        self.assertTrue(np.allclose(committor_sensitivity(self.T4, [0], [3], 1), self.qS41))      
        self.assertTrue(np.allclose(committor_sensitivity(self.T4, [0], [3], 2), self.qS42))      
        self.assertTrue(np.allclose(committor_sensitivity(self.T4, [0], [3], 3), self.S4zero))      
        
    def test_mfpt_sensitivity(self):
        
        self.assertTrue(np.allclose(mfpt_sensitivity(self.T4, 0, 0), self.S4zero))    
        self.assertTrue(np.allclose(mfpt_sensitivity(self.T4, 0, 1), self.mS01))      
        self.assertTrue(np.allclose(mfpt_sensitivity(self.T4, 0, 2), self.mS02))      
        self.assertTrue(np.allclose(mfpt_sensitivity(self.T4, 3, 2), self.mS32))  
        
    def test_eigenvector_sensitivity(self):
                        
        self.assertTrue(np.allclose(eigenvector_sensitivity(self.T4, 1 , 1), self.mV11, atol=1e-5))      
        self.assertTrue(np.allclose(eigenvector_sensitivity(self.T4, 2 , 2), self.mV22, atol=1e-5))      
        self.assertTrue(np.allclose(eigenvector_sensitivity(self.T4, 0 , 3), self.mV03, atol=1e-5))      

if __name__=="__main__":
    unittest.main()