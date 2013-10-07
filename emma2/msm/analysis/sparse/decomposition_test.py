"""Test package for the decomposition module"""
import unittest

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse

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
        """Sort eigenvalues and eigenvectors by increasing absolute value"""
        ind=np.argsort(abs(v))
        v=v[ind]
        L=L[:, ind]
        R=R[:, ind]
        
        nu=L[:,-1]
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
        mu_n=decomposition.mu(self.T_sparse)
        self.assertTrue(np.allclose(self.mu_sparse, mu_n))

    def test_eigenvalues(self):
        vn=decomposition.eigenvalues(self.T_sparse, k=self.k)
        ind=np.argsort(np.abs(vn))
        vn=vn[ind]
        self.assertTrue(np.allclose(vn, self.v_sparse))

    def test_eigenvectors(self):
        """Right eigenvectors"""        
        vn, Rn=decomposition.eigenvectors(self.T_sparse, k=self.k)

        """Sort eigenvalues and eigenvectors by absolute value"""
        ind=np.argsort(np.abs(vn))
        vn=vn[ind]
        Rn=Rn[:,ind]

        L_dense=self.L_sparse.toarray()

        """Compute overlapp between true left and computed right eigenvectors"""
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))
        

        """Left eigenvectors"""        
        vn, Ln=decomposition.eigenvectors(self.T_sparse, k=self.k, right=False)

        """Sort eigenvalues and eigenvectors by absolute value"""
        ind=np.argsort(np.abs(vn))
        vn=vn[ind]
        Ln=Ln[:,ind]

        R_dense=self.R_sparse.toarray()

        """Compute overlapp between true right and computed left eigenvectors"""
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))      

if __name__=="__main__":
    unittest.main()

    
    
    
    
    

