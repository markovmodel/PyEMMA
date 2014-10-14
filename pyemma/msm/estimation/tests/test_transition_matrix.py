import unittest
import warnings

import numpy as np
import scipy.sparse

from pyemma.msm.estimation import transition_matrix, tmatrix_cov, error_perturbation

"""Unit tests for the transition_matrix module"""

class TestTransitionMatrixNonReversibleSparse(unittest.TestCase):
    
    def setUp(self):
        """Small test cases"""
        self.C1=scipy.sparse.csr_matrix([[1, 3], [3, 1]])
        self.C2=scipy.sparse.csr_matrix([[0, 2], [1, 1]])
        
        self.T1=scipy.sparse.csr_matrix([[0.25, 0.75], [0.75, 0.25]])
        self.T2=scipy.sparse.csr_matrix([[0, 1], [0.5, 0.5]])
        
        """Zero row sum throws an error"""
        self.C0=scipy.sparse.csr_matrix([[0, 0], [3, 1]])

    def tearDown(self):
        pass

    def test_transition_matrix(self):
        """Small test cases"""
        T=transition_matrix(self.C1).toarray()
        self.assertTrue(np.allclose(T, self.T1.toarray()))
        
        T=transition_matrix(self.C1).toarray()
        self.assertTrue(np.allclose(T, self.T1.toarray()))        

class TestCovariance(unittest.TestCase):
    
    def setUp(self):
        alpha1=np.array([1.0, 2.0, 1.0])
        cov1=1.0/80*np.array([[ 3.0, -2.0, -1.0],  [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2=np.array([2.0, 1.0, 2.0])
        cov2=1.0/150*np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C=np.zeros((3, 3))
        self.C[0, :]=alpha1-1.0
        self.C[1, :]=alpha2-1.0
        self.C[2, :]=alpha1-1.0

        self.cov=np.zeros((3, 3, 3))
        self.cov[0, :, :]=cov1
        self.cov[1, :, :]=cov2
        self.cov[2, :, :]=cov1
        
    def tearDown(self):
        pass

    def test_tmatrix_cov(self):
        cov=tmatrix_cov(self.C)
        self.assertTrue(np.allclose(cov, self.cov))

        cov=tmatrix_cov(self.C, k=1)
        self.assertTrue(np.allclose(cov, self.cov[1, :, :]))

class TestErrorPerturbation(unittest.TestCase):
    def setUp(self):
        alpha1=np.array([1.0, 2.0, 1.0])
        cov1=1.0/80*np.array([[ 3.0, -2.0, -1.0],  [-2.0, 4.0, -2.0], [-1.0, -2.0, 3.0]])

        alpha2=np.array([2.0, 1.0, 2.0])
        cov2=1.0/150*np.array([[6, -2, -4], [-2, 4, -2], [-4, -2, 6]])

        self.C=np.zeros((3, 3))
        self.C[0, :]=alpha1-1.0
        self.C[1, :]=alpha2-1.0
        self.C[2, :]=alpha1-1.0

        self.cov=np.zeros((3, 3, 3))
        self.cov[0, :, :]=cov1
        self.cov[1, :, :]=cov2
        self.cov[2, :, :]=cov1

        """Scalar-valued observable f(P)=P_11+P_22+P_33"""
        self.S1=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        """Vector-valued observable f(P)=(P_11, P_12)"""
        self.S2=np.array([ [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])
        
        """Error-perturbation scalar observable"""
        self.x=(self.S1[:,:,np.newaxis]*self.cov*self.S1[:,np.newaxis,:]).sum()
  
        """Error-perturbation vector observable"""
        tmp=self.S2[:,np.newaxis,:,:,np.newaxis]*self.cov[np.newaxis,np.newaxis,:,:,:]*\
            self.S2[np.newaxis,:,:,np.newaxis,:]
        self.X=np.sum(tmp, axis=(2,3,4))

    def tearDown(self):
        pass

    def test_error_perturbation(self):
        xn=error_perturbation(self.C, self.S1)
        self.assertTrue(np.allclose(xn, self.x))

        Xn=error_perturbation(self.C, self.S2)
        self.assertTrue(np.allclose(Xn, self.X))   

    def test_error_perturbation_sparse(self):
        Csparse=scipy.sparse.csr_matrix(self.C)

        with warnings.catch_warnings(record=True) as w:
            xn=error_perturbation(Csparse,self.S1)
            self.assertTrue(np.allclose(xn, self.x))

        with warnings.catch_warnings(record=True) as w:
            Xn=error_perturbation(Csparse,self.S2)
            self.assertTrue(np.allclose(Xn, self.X))
        
        
if __name__=="__main__":
    unittest.main()
