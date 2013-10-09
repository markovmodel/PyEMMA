"""This module provides unit tests for the expectations module"""

import unittest
import numpy as np
from scipy.linalg import eig

import expectations

class TestExpectations(unittest.TestCase):
    def setUp(self):
        self.dim=100
        C=np.random.random_integers(0, 50, size=(self.dim, self.dim))
        C=0.5*(C+np.transpose(C))
        self.T=C/np.sum(C, axis=1)[:, np.newaxis]
        """Eigenvalues and left eigenvectors, sorted"""
        v, L=eig(np.transpose(self.T))
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]
        L=L[:,ind]
        """Compute stationary distribution"""
        self.mu=L[:, 0]/np.sum(L[:, 0])
        
        pass
    def tearDown(self):
        pass

    def test_expected_counts(self):
        p0=self.mu
        T=self.T

        N=20
        EC_n=expectations.expected_counts(p0, T, N)

        """
        If p0 is the stationary vector the computation can
        be carried out by a simple multiplication
        """        
        EC_true=N*self.mu[:,np.newaxis]*T
        self.assertTrue(np.allclose(EC_true, EC_n))

        """Zero length chain"""
        N=0
        EC_n=expectations.expected_counts(p0, T, N)
        EC_true=np.zeros(T.shape)
        self.assertTrue(np.allclose(EC_true, EC_n))
        
if __name__=="__main__":
    unittest.main()

        
