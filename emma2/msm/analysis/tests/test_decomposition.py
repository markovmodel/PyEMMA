r"""Unit test for decomposition functions in api.py

.. moduleauthor:: Benjamin Trendelkamp-Schroer<benjamin DOT trendelkamp-schorer AT fu-berlin DOT de>

"""
import sys

import unittest
import warnings

import numpy as np

import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from scipy.linalg import eig, eigh, eigvals, eigvalsh, qr, solve

from emma2.util.exceptions import SpectralWarning

from emma2.msm.analysis import stationary_distribution, eigenvalues, eigenvectors, rdl_decomposition, timescales

from birth_death_chain import BirthDeathChain

################################################################################
# Dense
################################################################################

def random_orthorgonal_matrix(d):
    r"""Compute a random orthorgonal matrix.

    The algorithm proceeds in two steps
    i) Generate a dxd square matrix with random normal distributed entries
    ii) Use a QR factorization to obtain a random orthorgonal matrix Q

    """
    X=np.random.randn(d, d)
    Q, R=qr(X)
    return Q

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

class TestDecompositionDense(unittest.TestCase):
    
    def setUp(self):
        self.dim=100

        """Random left eigenvectors"""
        self.L=random_linearly_independent_vectors(self.dim, self.dim)
        """Corresponding right eigenvectors"""
        self.R=solve(np.transpose(self.L), np.eye(self.dim))    
        # """Random eigenvalues uniform in (0, 1)"""
        # v=random.rand(self.dim)
        v=np.linspace(0.1, 1.0, self.dim)

        """
        Order eigenvalues by absolute value - 
        this ordering is the desired behaviour
        for decomposition.eigenvalues.
        """
        ind=np.argsort(np.abs(v))[::-1]
        self.v=v[ind]

        """Assemble test matrix A=RDL^T"""
        self.A=np.dot(self.R, np.dot(np.diag(self.v), np.transpose(self.L)))

    def tearDown(self):
        pass

    def test_stationary_distribution(self): 
        """Set up bdc-chain"""
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1
        bdc=BirthDeathChain(q, p)
       
        """Transition matrix"""
        T=bdc.transition_matrix()
        """Stationary distribution"""
        statdist_true=bdc.stationary_distribution()
        
        """Compute stationary distribution"""
        statdist_test=stationary_distribution(T)

        """Assert numerical equality of statdist_true and statdist_test"""
        self.assertTrue(np.allclose(statdist_test, statdist_true))

    def test_eigenvalues(self):
        """Compute eigenvalues"""
        vn=eigenvalues(self.A)

        """Assert numerical equality of vn an v"""
        self.assertTrue(np.allclose(self.v, vn))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        vn=eigenvalues(self.A, k=ind)
        self.assertTrue(np.allclose(self.v[np.asarray(ind)], vn))

        ind=[0, 2, 7] # list
        vn=eigenvalues(self.A, k=ind)
        self.assertTrue(np.allclose(self.v[np.asarray(ind)], vn))
        
        # ind=array([0, 2, 7]) # ndarray
        # vn=decomposition.eigenvalues(A, k=ind)
        # self.assertTrue(allclose(v[ind], vn))

        k=5 # number of eigenvalues
        ind=np.arange(k)
        vn=eigenvalues(self.A, k=k)
        self.assertTrue(np.allclose(self.v[ind], vn))
     
    def test_eigenvectors(self):
        """Compute right eigenvectors"""
        Rn=eigenvectors(self.A)
        
        """Overlapp matrix X of computed eigenvectors Rn
           and true left eigenvectors L"""
        X=np.dot(np.transpose(self.L), Rn)

        """Should be zero for all elements except the diagonal"""
        ind_diag=np.diag_indices(self.dim)
        """Set diagonal indices to zero"""
        X[ind_diag]=0.0    

        """Check that X is approx 0"""
        self.assertTrue(np.allclose(X, 0.0))

        """Now for left eigenvectors"""
        Ln=eigenvectors(self.A, right=False)
        
        """Overlapp matrix"""
        X=np.dot(np.transpose(Ln), self.R)
        """Set diagonal indices to zero"""
        X[ind_diag]=0.0    

        """Check that X is approx 0"""
        self.assertTrue(np.allclose(X, 0.0))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        ind_diag=np.diag_indices(len(ind))        
        Rn=eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        ind=[0, 2, 7] # list
        ind_diag=np.diag_indices(len(ind))        
        Rn=eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        ind=np.array([0, 2, 7]) # ndarray
        ind_diag=np.diag_indices(len(ind))        
        Rn=eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        k=5 # number of eigenvalues
        ind=np.arange(k)
        ind_diag=np.diag_indices(len(ind))        
        Rn=eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

    def test_rdl_decomposition(self):
        Rn, Dn, Ln=rdl_decomposition(self.A)
        vn=np.diagonal(Dn)

        """Eigenvalues"""
        self.assertTrue(np.allclose(self.v, vn))

        """Eigenvectors"""
        ind_diag=np.diag_indices(self.dim)
        
        X=np.dot(np.transpose(self.L), Rn)
        X[ind_diag]=0.0
        self.assertTrue(np.allclose(X, 0.0))

        # X=np.dot(np.transpose(Ln), self.R)
        # X[ind_diag]=0.0
        # self.assertTrue(np.allclose(X, 0.0))

        """
        The computed left eigenvectors Ln are sometimes 
        not perfectly orthogonal to the true right eienvectors
        R. This is why the above test does sometimes fail.
        """
        
        """They are however 'good' left eigenvectors for A"""
        X=np.dot(Ln, self.A)
        self.assertTrue(np.allclose(X, vn[:, np.newaxis]*Ln))

    def test_timescales(self):
        ts_n=timescales(self.A)
        ts=np.zeros(len(self.v))
        ts[0]=np.inf
        ts[1:]=-1.0/np.log(self.v[1:])
        self.assertTrue(np.allclose(ts, ts_n))

        v=np.random.rand(self.dim)
        v[0:5]=1.0
        
class TestTimescalesDense(unittest.TestCase):

    def setUp(self):
        self.dim=100

        """Random left eigenvectors"""
        self.L=random_linearly_independent_vectors(self.dim, self.dim)
        """Corresponding right eigenvectors"""
        self.R=solve(np.transpose(self.L), np.eye(self.dim))    

        """Purely real spectrum, unique eigenvalue with modulus one"""
        v_real=np.linspace(0.1, 1.0, self.dim)
        ind=np.argsort(np.abs(v_real))[::-1]
        self.v_real=v_real[ind]
        self.A_real=np.dot(self.R, np.dot(np.diag(self.v_real), np.transpose(self.L)))
        self.ts_real=np.zeros(len(self.v_real))
        self.ts_real[0]=np.inf
        self.ts_real[1:]=-1.0/np.log(np.abs(self.v_real[1:]))

        """Complex spectrum, unique eigenvalue with modulus one"""
        v_complex=np.linspace(0.1, 1.0, self.dim)+0.0*1j
        v_complex[1:5]=0.9+0.1*1j
        ind=np.argsort(np.abs(v_complex))[::-1]
        self.v_complex=v_complex[ind]
        self.A_complex=np.dot(self.R, np.dot(np.diag(self.v_complex), np.transpose(self.L)))
        self.ts_complex=np.zeros(len(self.v_complex))
        self.ts_complex[0]=np.inf
        self.ts_complex[1:]=-1.0/np.log(np.abs(self.v_complex[1:]))

        """Purely real spectrum, multiple eigenvalues with modulus one"""
        v_real_m=np.linspace(0.1, 1.0, self.dim)
        ind=np.argsort(np.abs(v_real_m))[::-1]
        self.v_real_m=v_real_m[ind]
        self.v_real_m[1:5]=1.0
        self.A_real_m=np.dot(self.R, np.dot(np.diag(self.v_real_m), np.transpose(self.L)))
        self.ts_real_m=np.zeros(len(self.v_real_m))
        self.ts_real_m[0:5]=np.inf
        self.ts_real_m[5:]=-1.0/np.log(np.abs(self.v_real_m[5:]))

        """Complex spectrum, multiple eigenvalues with modulus one"""
        v_complex_m=np.linspace(0.1, 1.0, self.dim)+0.0*1j
        ind=np.argsort(np.abs(v_complex_m))[::-1]
        self.v_complex_m=v_complex_m[ind]
        self.v_complex_m[1:5]=(1.0+1.0*1j)/np.sqrt(2.0)
        self.A_complex_m=np.dot(self.R, np.dot(np.diag(self.v_complex_m), np.transpose(self.L)))
        self.ts_complex_m=np.zeros(len(self.v_complex_m))
        self.ts_complex_m[0:5]=np.inf
        self.ts_complex_m[5:]=-1.0/np.log(np.abs(self.v_complex_m[5:]))      

       
    def tearDown(self):
        pass

    def mdot(self, *args):
        return reduce(np.dot, args)
    
    def test_timescales(self):
        """tau=1"""
        ts_n=timescales(self.A_real)
        self.assertTrue(np.allclose(ts_n, self.ts_real))
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex)
            self.assertTrue(np.allclose(ts_n, self.ts_complex))
            assert issubclass(w[-1].category, SpectralWarning)

        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_real_m)
            self.assertTrue(np.allclose(ts_n, self.ts_real_m))
            assert issubclass(w[-1].category, SpectralWarning)
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex_m)
            self.assertTrue(np.allclose(ts_n, self.ts_complex_m))
            assert issubclass(w[-1].category, SpectralWarning)

        """tau=10"""
        ts_n=timescales(self.A_real, tau=10)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real))

        """tau=10, k=8"""
        ts_n=timescales(self.A_real, tau=10, k=8)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:8]))

################################################################################
# Sparse
################################################################################

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

class TestDecompositionSparse(unittest.TestCase):
    
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

    def test_stationary_distribution(self):
        p=np.zeros(100)
        q=np.zeros(100)
        p[0:-1]=0.5
        q[1:]=0.5
        p[49]=0.01
        q[51]=0.1
        bdc=BirthDeathChain(q, p)           

        T=bdc.transition_matrix_sparse()
        mu=bdc.stationary_distribution()
        
        mu_n=stationary_distribution(T)
        self.assertTrue(np.allclose(mu, mu_n))

    def test_eigenvalues(self):
        vn=eigenvalues(self.T_sparse, k=self.k)        
        self.assertTrue(np.allclose(vn, self.v_sparse))

        """Test ncv keyword computing self.k/4 eigenvalues 
           with Kyrlov subspace of dim=self.k"""
        vn=eigenvalues(self.T_sparse, k=self.k/4, ncv=self.k)
        self.assertTrue(np.allclose(vn, self.v_sparse[0:self.k/4]))


    def test_eigenvectors(self):
        """Right eigenvectors"""        
        Rn=eigenvectors(self.T_sparse, k=self.k)        

        L_dense=self.L_sparse.toarray()

        """Compute overlapp between true left and computed right eigenvectors"""
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

        """Left eigenvectors"""        
        Ln=eigenvectors(self.T_sparse, k=self.k, right=False)
        R_dense=self.R_sparse.toarray()

        """Compute overlapp between true right and computed left eigenvectors"""
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

        """Check the same for self.k/4 eigenvectors wih ncv=k"""

        """Right eigenvectors"""        
        Rn=eigenvectors(self.T_sparse, k=self.k/4, ncv=self.k)        

        L_dense=self.L_sparse.toarray()[:,0:self.k/4]

        """Compute overlapp between true left and computed right eigenvectors"""
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

        """Left eigenvectors"""        
        Ln=eigenvectors(self.T_sparse, k=self.k/4, right=False, ncv=self.k)
        R_dense=self.R_sparse.toarray()[:,0:self.k/4]
        

        """Compute overlapp between true right and computed left eigenvectors"""
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))                                      

    def test_rdl_decomposition(self):
        # k=self.k
        # v, R=scipy.sparse.linalg.eigs(self.T_sparse, k=k, which='LM')
        # r, L=scipy.sparse.linalg.eigs(self.T_sparse.transpose(), k=k, which='LM')

        """Standard norm"""
        # vn, Ln, Rn=decomposition.rdl_decomposition(self.T_sparse, k=self.k)
        Rn, Dn, Ln=rdl_decomposition(self.T_sparse, k=self.k)
        vn=np.diagonal(Dn)
        Ln=np.transpose(Ln)

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
        # vn, Ln, Rn=decomposition.rdl_decomposition(self.T_sparse, k=self.k, norm='reversible')
        Rn, Dn, Ln=rdl_decomposition(self.T_sparse, k=self.k, norm='reversible')
        vn=np.diagonal(Dn)
        Ln=np.transpose(Ln)

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

        """Check the same for self.k/4 eigenvectors wih ncv=k"""
        
        """Standard norm"""
        Rn, Dn, Ln=rdl_decomposition(self.T_sparse, k=self.k/4, ncv=self.k)
        vn=np.diagonal(Dn)
        Ln=np.transpose(Ln)
        
        """Eigenvalues"""
        self.assertTrue(np.allclose(self.v_sparse[0:self.k/4], vn))

        """Computed left eigenvectors Ln"""
        R_dense=self.R_sparse.toarray()[:,0:self.k/4]
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

        """Computed right eigenvectors Rn"""
        L_dense=self.L_sparse.toarray()[:,0:self.k/4]        
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))        

        """Reversible"""
        Rn, Dn, Ln=rdl_decomposition(self.T_sparse, k=self.k/4,\
                                                       norm='reversible', ncv=self.k)
        vn=np.diagonal(Dn)
        Ln=np.transpose(Ln)

        """Eigenvalues"""
        self.assertTrue(np.allclose(self.v_sparse[0:self.k/4], vn))

        """Computed left eigenvectors Ln"""
        R_dense=self.R_sparse.toarray()[:,0:self.k/4]
        A=np.dot(np.transpose(Ln), R_dense)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

        """Computed right eigenvectors Rn"""
        L_dense=self.L_sparse.toarray()[:,0:self.k/4]        
        A=np.dot(np.transpose(L_dense), Rn)
        ind_diag=np.diag_indices(self.k/4)
        A[ind_diag]=0.0

        """Assert that off-diagonal elements are zero"""
        self.assertTrue(np.allclose(A, 0.0))

class TestTimescalesSparse(unittest.TestCase):

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
        return reduce(np.dot, args)
    
    def test_timescales(self):
        """tau=1"""
        ts_n=timescales(self.A_real, k=self.k)
        self.assertTrue(np.allclose(ts_n, self.ts_real))
        # enable all warnings
        warnings.simplefilter("always")
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_complex))
            assert issubclass(w[-1].category, SpectralWarning)

        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_real_m, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_real_m))
            assert issubclass(w[-1].category, SpectralWarning)
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex_m, k=self.k)
            self.assertTrue(np.allclose(ts_n, self.ts_complex_m))
            assert issubclass(w[-1].category, SpectralWarning)

        """tau=10"""
        ts_n=timescales(self.A_real, tau=10, k=self.k)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real))

        """tau=10, k=8"""
        ts_n=timescales(self.A_real, tau=10, k=8)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:8]))

        """Same with k=self.k/4 and ncv=self.k"""

        """tau=1"""
        ts_n=timescales(self.A_real, k=self.k/4, ncv=self.k)
        # print ts_n
        # print self.ts_real[0:self.k/4]
        self.assertTrue(np.allclose(ts_n, self.ts_real[0:self.k/4]))        
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex, k=self.k/4, ncv=self.k)
            # print ts_n
            # print self.ts_complex[0:self.k/4]
            self.assertTrue(np.allclose(ts_n, self.ts_complex[0:self.k/4]))
            assert issubclass(w[-1].category, SpectralWarning)

        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_real_m, k=self.k/4, ncv=2*self.k)
            # print ts_n
            # print self.ts_real_m[0:self.k/4]
            self.assertTrue(np.allclose(ts_n, self.ts_real_m[0:self.k/4]))
            assert issubclass(w[-1].category, SpectralWarning)

        with warnings.catch_warnings(record=True) as w:
            ts_n=timescales(self.A_complex_m, k=self.k/4, ncv=self.k)
            # print ts_n
            # print self.ts_complex_m[0:self.k/4]
            self.assertTrue(np.allclose(ts_n, self.ts_complex_m[0:self.k/4]))
            assert issubclass(w[-1].category, SpectralWarning)

        """tau=10"""
        ts_n=timescales(self.A_real, tau=10, k=self.k/4, ncv=self.k)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:self.k/4]))

        """tau=10, k=8"""
        ts_n=timescales(self.A_real, tau=10, k=8, ncv=self.k)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:8]))
        
        
if __name__=="__main__":
    unittest.main()
