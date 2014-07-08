r"""This module provides unit tests for the expectations module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from scipy.linalg import eig

import expectations

class TestExpectedCounts(unittest.TestCase):
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

        N=2000
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

class TestExpectedCountsStationary(unittest.TestCase):
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
        
    def tearDown(self):
        pass

    def test_expected_counts_stationary(self):
        T=self.T
        N=20

        """Compute mu on the fly"""
        EC_n=expectations.expected_counts_stationary(T, N)
        EC_true=N*self.mu[:,np.newaxis]*T
        self.assertTrue(np.allclose(EC_true, EC_n))

        """Use precomputed mu"""
        EC_n=expectations.expected_counts_stationary(T, N, mu=self.mu)
        EC_true=N*self.mu[:,np.newaxis]*T
        self.assertTrue(np.allclose(EC_true, EC_n))
    
        

class TestEcMatrixVector(unittest.TestCase):
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

    def test_ec_matrix_vector(self):
        p0=self.mu
        T=self.T

        N=20
        EC_n=expectations.ec_matrix_vector(p0, T, N)

        """
        If p0 is the stationary vector the computation can
        be carried out by a simple multiplication
        """        
        EC_true=N*self.mu[:,np.newaxis]*T
        self.assertTrue(np.allclose(EC_true, EC_n))

        """Zero length chain"""
        N=0
        EC_n=expectations.ec_matrix_vector(p0, T, N)
        EC_true=np.zeros(T.shape)
        self.assertTrue(np.allclose(EC_true, EC_n))

class TestEcGeometricSeries(unittest.TestCase):
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

    def test_ec_geometric_series(self):
        p0=self.mu
        T=self.T

        N=2000
        EC_n=expectations.ec_geometric_series(p0, T, N)

        """
        If p0 is the stationary vector the computation can
        be carried out by a simple multiplication
        """        
        EC_true=N*self.mu[:,np.newaxis]*T
        self.assertTrue(np.allclose(EC_true, EC_n))

        """Zero length chain"""
        N=0
        EC_n=expectations.ec_geometric_series(p0, T, N)
        EC_true=np.zeros(T.shape)
        self.assertTrue(np.allclose(EC_true, EC_n))

class TestGeometricSeries(unittest.TestCase):
    def setUp(self):
        self.q=2.0
        self.q_array=np.array([2.0, 1.0, 0.8, -0.3, -1.0, -2.0])
        self.n=9

        self.s=0        
        for i in range(self.n+1):
            self.s+=self.q**i

        self.s_array=np.zeros(self.q_array.shape)
        for i in range(self.n+1):
            self.s_array+=self.q_array**i
        

    def tearDown(self):
        pass

    def test_geometric_series(self):
        x=expectations.geometric_series(self.q, self.n)     
        self.assertTrue(np.allclose(x, self.s))

        x=expectations.geometric_series(self.q_array, self.n)        
        self.assertTrue(np.allclose(x, self.s_array))

        """Assert ValueError for negative n"""
        with self.assertRaises(ValueError):
            expectations.geometric_series(self.q, -2)

        with self.assertRaises(ValueError):
            expectations.geometric_series(self.q_array, -2)
     
        
if __name__=="__main__":
    unittest.main()

        
