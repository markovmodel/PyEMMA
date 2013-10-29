"""Test package for the decomposition module"""
import unittest

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import decomposition

def random_orthonormal_sparse_vectors(d, k):
    r"""Generate a random set of k orthonormal sparse vectors 

    The algorithm draws random indices, {i_1,...,i_k}, from the set
    of all possible indices, {0,...,d-1}, without replacement.
    Random sparse vectors v are given by

    v[i]=k^{-1/2} for i in {i_1,...,i_k} and zero elsewhere.

    """
    indices=np.random.choice(d, replace=False, size=(k*k))
    indptr=np.arange(0, k*(k+1), k)
    values=1.0/np.sqrt(k)*np.ones(k*k)
    return scipy.sparse.csc_matrix((values, indices, indptr))    


class TestDecomposition(unittest.TestCase):
    
    def setUp(self):
        self.k=20
        self.d=10000
        """Generate a random kxk dense transition matrix"""
        C=np.random.random_integers(0, 100, size=(self.k, self.k))
        C=C+np.transpose(C) # Symmetric count matrix for real eigenvalues
        T=1.0*C/np.sum(C, axis=1)[:, np.newaxis]
        v, L, R=scipy.linalg.eig(T, left=True, right=True)
        """Sort eigenvalues and eigenvectors, order is decreasing absolute value"""
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]
        L=L[:, ind]
        R=R[:, ind]
        
        nu=L[:, 0]
        mu=nu/np.sum(nu)

        """
        Generate k random sparse
        orthorgonal vectors of dimension d
        """
        Q=random_orthonormal_sparse_vectors(self.d, self.k)
        
        """Push forward dense decomposition to sparse one via Q"""
        self.L_sparse=Q.dot(scipy.sparse.csr_matrix(L))
        self.R_sparse=Q.dot(scipy.sparse.csr_matrix(R))
        self.v_sparse=v # Eigenvalues are invariant

        """Push forward transition matrix and stationary distribution"""
        self.T_sparse=Q.dot(scipy.sparse.csr_matrix(T)).dot(Q.transpose())
        self.mu_sparse=Q.dot(mu)/np.sqrt(self.k)

    def tearDown(self):
        pass

    def test_mu(self):           
        mu_n=decomposition.stationary_distribution(self.T_sparse)
        self.assertTrue(np.allclose(self.mu_sparse, mu_n))

    def test_eigenvalues(self):
        vn=decomposition.eigenvalues(self.T_sparse, k=self.k)        
        self.assertTrue(np.allclose(vn, self.v_sparse))

    def test_eigenvectors(self):
        """Right eigenvectors"""        
        Rn=decomposition.eigenvectors(self.T_sparse, k=self.k)        

        L_dense=self.L_sparse.toarray()

        """Compute overlapp between true left and computed right eigenvectors"""
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

        """Left eigenvectors"""        
        Ln=decomposition.eigenvectors(self.T_sparse, k=self.k, right=False)
        R_dense=self.R_sparse.toarray()

        """Compute overlapp between true right and computed left eigenvectors"""
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

    def test_rdl_decomposition(self):
        # k=self.k
        # v, R=scipy.sparse.linalg.eigs(self.T_sparse, k=k, which='LM')
        # r, L=scipy.sparse.linalg.eigs(self.T_sparse.transpose(), k=k, which='LM')

        """Standard norm"""
        vn, Ln, Rn=decomposition.rdl_decomposition(self.T_sparse, k=self.k)

        """Eigenvalues"""
        self.assertTrue(np.allclose(self.v_sparse, vn))

        """Computed left eigenvectors Ln"""
        R_dense=self.R_sparse.toarray()
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

        """Computed right eigenvectors Rn"""
        L_dense=self.L_sparse.toarray()        
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

        """Reversible"""
        vn, Ln, Rn=decomposition.rdl_decomposition(self.T_sparse, k=self.k, norm='reversible')

        """Eigenvalues"""
        self.assertTrue(np.allclose(self.v_sparse, vn))

        """Computed left eigenvectors Ln"""
        R_dense=self.R_sparse.toarray()
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

        """Computed right eigenvectors Rn"""
        L_dense=self.L_sparse.toarray()        
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

if __name__=="__main__":
    unittest.main()

    
    
    
    
    

