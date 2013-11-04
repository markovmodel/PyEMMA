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


def random_linearly_independent_vectors(d, k):
    r"""Generate a set of linear independent vectors
    
    The algorithm picks k random points uniformly distributed
    on the d-sphere. They will form a set of linear independent
    vectors with probability one for k<=d.

    """
    if k>d:
        raise ValueError("Can not pick more linear independent vectors"+\
                             " than the full dimension of the vector space")
    else:
        """Pick k-vectors with gaussian distributed entries"""
        G=np.random.randn(d, k)
        """Normalize to length=1 to get uniform distribution on the d-sphere"""
        length=np.sqrt(np.sum(G**2, axis=0))
        X=G/length[np.newaxis, :]
        return X

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

class TestTimescales(unittest.TestCase):

    def setUp(self):
        self.d=10000
        self.k=20

        """Random sparse orthonormal vectors for blow-up"""
        Q=random_orthonormal_sparse_vectors(self.d, self.k)

        """Random left eigenvectors"""
        self.L=random_linearly_independent_vectors(self.k, self.k)
        """Corresponding right eigenvectors"""
        self.R=np.linalg.solve(np.transpose(self.L), np.eye(self.k))    

        """Purely real spectrum, unique eigenvalue with modulus one"""
        v_real=np.linspace(0.1, 1.0, self.k)
        ind=np.argsort(np.abs(v_real))[::-1]
        self.v_real=v_real[ind]
        A=np.dot(self.R, np.dot(np.diag(self.v_real), np.transpose(self.L)))
        self.A_real=Q.dot(scipy.sparse.csr_matrix(A)).dot(Q.transpose())        
        self.ts_real=np.zeros(len(self.v_real))
        self.ts_real[0]=np.inf
        self.ts_real[1:]=-1.0/np.log(np.abs(self.v_real[1:]))

        """Complex spectrum, unique eigenvalue with modulus one"""
        v_complex=np.linspace(0.1, 1.0, self.k)+0.0*1j
        v_complex[1:5]=0.9+0.1*1j
        ind=np.argsort(np.abs(v_complex))[::-1]
        self.v_complex=v_complex[ind]
        A=np.dot(self.R, np.dot(np.diag(self.v_complex), np.transpose(self.L)))
        self.A_complex=Q.dot(scipy.sparse.csr_matrix(A)).dot(Q.transpose())        
        self.ts_complex=np.zeros(len(self.v_complex))
        self.ts_complex[0]=np.inf
        self.ts_complex[1:]=-1.0/np.log(np.abs(self.v_complex[1:]))

        """Purely real spectrum, multiple eigenvalues with modulus one"""
        v_real_m=np.linspace(0.1, 1.0, self.k)
        ind=np.argsort(np.abs(v_real_m))[::-1]
        self.v_real_m=v_real_m[ind]
        self.v_real_m[1:5]=1.0
        A=np.dot(self.R, np.dot(np.diag(self.v_real_m), np.transpose(self.L)))
        self.A_real_m=Q.dot(scipy.sparse.csr_matrix(A)).dot(Q.transpose())        
        self.ts_real_m=np.zeros(len(self.v_real_m))
        self.ts_real_m[0:5]=np.inf
        self.ts_real_m[5:]=-1.0/np.log(np.abs(self.v_real_m[5:]))

        """Complex spectrum, multiple eigenvalues with modulus one"""
        v_complex_m=np.linspace(0.1, 1.0, self.k)+0.0*1j
        ind=np.argsort(np.abs(v_complex_m))[::-1]
        self.v_complex_m=v_complex_m[ind]
        self.v_complex_m[1:5]=(1.0+1.0*1j)/np.sqrt(2.0)
        A=np.dot(self.R, np.dot(np.diag(self.v_complex_m), np.transpose(self.L)))
        self.A_complex_m=Q.dot(scipy.sparse.csr_matrix(A)).dot(Q.transpose())        
        self.ts_complex_m=np.zeros(len(self.v_complex_m))
        self.ts_complex_m[0:5]=np.inf
        self.ts_complex_m[5:]=-1.0/np.log(np.abs(self.v_complex_m[5:]))      

       
    def tearDown(self):
        pass

    def mdot(self, *args):
        return reduce(numpy.dot, args)
    
    def test_timescales(self):
        """tau=1"""
        ts_n=decomposition.timescales(self.A_real, k=self.k)
        self.assertTrue(np.allclose(ts_n, self.ts_real))
        
        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_complex, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_complex))

        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_real_m, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_real_m))
        
        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_complex_m, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_complex_m))           

        """tau=10"""
        ts_n=decomposition.timescales(self.A_real, tau=10, k=self.k)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real))

        """tau=10, k=8"""
        ts_n=decomposition.timescales(self.A_real, tau=10, k=8)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:8]))
        
if __name__=="__main__":
    unittest.main()

    
    
    
    
    

