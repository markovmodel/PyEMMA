r"""This module provides unit tests for the decomposition module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import warnings

import numpy as np

from scipy.linalg import eig, eigh, eigvals, eigvalsh, qr, solve

from emma2.util.exceptions import SpectralWarning
import decomposition

from committor_test import BirthDeathChain

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

class TestDecomposition(unittest.TestCase):
    
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

    def test_mu_eigenvector(self):        
        """Random bin counts"""
        c=np.random.random_integers(0, 100, size=(self.dim,))
        """Dirichlet distributed probabilities"""
        statdist_true=np.random.dirichlet(c)
        """Transition matrix containing statdist_true in every row"""
        T=statdist_true[np.newaxis, :]*np.ones(self.dim)[:,np.newaxis]
        
        """Compute stationary distribution"""
        statdist_test=decomposition.stationary_distribution_from_eigenvector(T)

        """Assert numerical equality of statdist_true and statdist_test"""
        self.assertTrue(np.allclose(statdist_test, statdist_true))

    def test_mu_backward_iteration(self):        
        """Random bin counts"""
        c=np.random.random_integers(0, 100, size=(self.dim,))
        """Dirichlet distributed probabilities"""
        statdist_true=np.random.dirichlet(c)
        """Transition matrix containing statdist_true in every row"""
        T=statdist_true[np.newaxis, :]*np.ones(self.dim)[:,np.newaxis]
        
        """Compute stationary distribution"""
        statdist_test=decomposition.stationary_distribution_from_backward_iteration(T)

        """Assert numerical equality of statdist_true and statdist_test"""
        self.assertTrue(np.allclose(statdist_test, statdist_true))
        
    def test_eigenvalues(self):
        """Compute eigenvalues"""
        vn=decomposition.eigenvalues(self.A)

        """Assert numerical equality of vn an v"""
        self.assertTrue(np.allclose(self.v, vn))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        vn=decomposition.eigenvalues(self.A, k=ind)
        self.assertTrue(np.allclose(self.v[np.asarray(ind)], vn))

        ind=[0, 2, 7] # list
        vn=decomposition.eigenvalues(self.A, k=ind)
        self.assertTrue(np.allclose(self.v[np.asarray(ind)], vn))
        
        # ind=array([0, 2, 7]) # ndarray
        # vn=decomposition.eigenvalues(A, k=ind)
        # self.assertTrue(allclose(v[ind], vn))

        k=5 # number of eigenvalues
        ind=np.arange(k)
        vn=decomposition.eigenvalues(self.A, k=k)
        self.assertTrue(np.allclose(self.v[ind], vn))
     
    def test_eigenvectors(self):
        """Compute right eigenvectors"""
        Rn=decomposition.eigenvectors(self.A)
        
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
        Ln=decomposition.eigenvectors(self.A, right=False)
        
        """Overlapp matrix"""
        X=np.dot(np.transpose(Ln), self.R)
        """Set diagonal indices to zero"""
        X[ind_diag]=0.0    

        """Check that X is approx 0"""
        self.assertTrue(np.allclose(X, 0.0))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        ind_diag=np.diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        ind=[0, 2, 7] # list
        ind_diag=np.diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        ind=np.array([0, 2, 7]) # ndarray
        ind_diag=np.diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

        k=5 # number of eigenvalues
        ind=np.arange(k)
        ind_diag=np.diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=np.dot(np.transpose(self.L[:,np.asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(np.allclose(X, 0.0))

    def test_rdl_decomposition(self):
        """Standard norm"""
        Rn, Dn, Ln=decomposition.rdl_decomposition(self.A)
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

        """Reversible"""

        """Use reversible birth-death-chain for test"""
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1
        
        bdc=BirthDeathChain(q, p)
        
        mu = bdc.stationary_distribution()
        T = bdc.transition_matrix()

        """Eigenvalues to check against"""
        v=eigvals(T)
        ind=np.argsort(np.abs(v))[::-1]
        v=v[ind]

        Rn, Dn, Ln=decomposition.rdl_decomposition(T, norm='reversible')

        """Eigenvalues"""
        vn=np.diagonal(Dn)
        self.assertTrue(np.allclose(v, vn))

        """Orthogonality of numerical eigenvectors"""
        Xn=np.dot(Ln, Rn)
        self.assertTrue(np.allclose(Xn, np.eye(T.shape[0])))

        """Check that left eigenvectors can be generated from right ones"""
        self.assertTrue(np.allclose(Ln.transpose(), mu[:,np.newaxis]*Rn))
        
        """Eigenvectors"""
        self.assertTrue(np.allclose(np.dot(T, Rn), np.dot(Rn, Dn)))
        self.assertTrue(np.allclose(np.dot(Ln, T), np.dot(Dn, Ln)))

    def test_timescales(self):
        ts_n=decomposition.timescales(self.A)
        ts=np.zeros(len(self.v))
        ts[0]=np.inf
        ts[1:]=-1.0/np.log(self.v[1:])
        self.assertTrue(np.allclose(ts, ts_n))

        v=np.random.rand(self.dim)
        v[0:5]=1.0
        
class TestTimescales(unittest.TestCase):

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
        ts_n=decomposition.timescales(self.A_real)
        self.assertTrue(np.allclose(ts_n, self.ts_real))
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=decomposition.timescales(self.A_complex)
            self.assertTrue(np.allclose(ts_n, self.ts_complex))
            assert issubclass(w[-1].category, SpectralWarning)

        with warnings.catch_warnings(record=True) as w:
            ts_n=decomposition.timescales(self.A_real_m)
            self.assertTrue(np.allclose(ts_n, self.ts_real_m))
            assert issubclass(w[-1].category, SpectralWarning)
        
        with warnings.catch_warnings(record=True) as w:
            ts_n=decomposition.timescales(self.A_complex_m)
            self.assertTrue(np.allclose(ts_n, self.ts_complex_m))
            assert issubclass(w[-1].category, SpectralWarning)

        """tau=10"""
        ts_n=decomposition.timescales(self.A_real, tau=10)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real))

        """tau=10, k=8"""
        ts_n=decomposition.timescales(self.A_real, tau=10, k=8)
        self.assertTrue(np.allclose(ts_n, 10*self.ts_real[0:8]))
        
        
if __name__=="__main__":
    unittest.main()
        
