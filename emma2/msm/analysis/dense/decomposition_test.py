"""This module provides unit tests for the decomposition module"""

import unittest

import numpy
from numpy import random, sum, sqrt, newaxis, ones, allclose, eye, asarray, abs, linspace
from numpy import  diag, transpose, argsort, dot, asarray, array, arange, diag_indices
from numpy import conjugate, zeros, log, inf
from scipy.linalg import eig, eigh, eigvals, eigvalsh, qr, solve

import decomposition

def random_orthorgonal_matrix(d):
    r"""Compute a random orthorgonal matrix.

    The algorithm proceeds in two steps
    i) Generate a dxd square matrix with random normal distributed entries
    ii) Use a QR factorization to obtain a random orthorgonal matrix Q

    """
    X=random.randn(d, d)
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
        G=random.randn(d, k)
        """Normalize to length=1 to get uniform distribution on the d-sphere"""
        length=sqrt(sum(G**2, axis=0))
        X=G/length[newaxis, :]
        return X

class TestDecomposition(unittest.TestCase):
    
    def setUp(self):
        self.dim=100

        """Random left eigenvectors"""
        self.L=random_linearly_independent_vectors(self.dim, self.dim)
        """Corresponding right eigenvectors"""
        self.R=solve(transpose(self.L), eye(self.dim))    
        # """Random eigenvalues uniform in (0, 1)"""
        # v=random.rand(self.dim)
        v=linspace(0.1, 1.0, self.dim)

        """
        Order eigenvalues by absolute value - 
        this ordering is the desired behaviour
        for decomposition.eigenvalues.
        """
        ind=argsort(abs(v))[::-1]
        self.v=v[ind]

        """Assemble test matrix A=RDL^T"""
        self.A=dot(self.R, dot(diag(self.v), transpose(self.L)))

    def tearDown(self):
        pass

    def test_mu(self):        
        """Random bin counts"""
        c=random.random_integers(0, 100, size=(self.dim,))
        """Dirichlet distributed probabilities"""
        statdist_true=random.dirichlet(c)
        """Transition matrix containing statdist_true in every row"""
        T=statdist_true[newaxis, :]*ones(self.dim)[:,newaxis]
        
        """Compute stationary distribution"""
        statdist_test=decomposition.stationary_distribution(T)

        """Assert numerical equality of statdist_true and statdist_test"""
        self.assertTrue(allclose(statdist_test, statdist_true))
        
    def test_eigenvalues(self):
        """Compute eigenvalues"""
        vn=decomposition.eigenvalues(self.A)

        """Assert numerical equality of vn an v"""
        self.assertTrue(allclose(self.v, vn))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        vn=decomposition.eigenvalues(self.A, k=ind)
        self.assertTrue(allclose(self.v[asarray(ind)], vn))

        ind=[0, 2, 7] # list
        vn=decomposition.eigenvalues(self.A, k=ind)
        self.assertTrue(allclose(self.v[asarray(ind)], vn))
        
        # ind=array([0, 2, 7]) # ndarray
        # vn=decomposition.eigenvalues(A, k=ind)
        # self.assertTrue(allclose(v[ind], vn))

        k=5 # number of eigenvalues
        ind=arange(k)
        vn=decomposition.eigenvalues(self.A, k=k)
        self.assertTrue(allclose(self.v[ind], vn))
     
    def test_eigenvectors(self):
        """Compute right eigenvectors"""
        Rn=decomposition.eigenvectors(self.A)
        
        """Overlapp matrix X of computed eigenvectors Rn
           and true left eigenvectors L"""
        X=dot(transpose(self.L), Rn)

        """Should be zero for all elements except the diagonal"""
        ind_diag=diag_indices(self.dim)
        """Set diagonal indices to zero"""
        X[ind_diag]=0.0    

        """Check that X is approx 0"""
        self.assertTrue(allclose(X, 0.0))

        """Now for left eigenvectors"""
        Ln=decomposition.eigenvectors(self.A, right=False)
        
        """Overlapp matrix"""
        X=dot(transpose(Ln), self.R)
        """Set diagonal indices to zero"""
        X[ind_diag]=0.0    

        """Check that X is approx 0"""
        self.assertTrue(allclose(X, 0.0))

        """Check behaviour for k-keyword"""
        ind=(0, 2, 7) # tuple
        ind_diag=diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=dot(transpose(self.L[:,asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(allclose(X, 0.0))

        ind=[0, 2, 7] # list
        ind_diag=diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=dot(transpose(self.L[:,asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(allclose(X, 0.0))

        ind=array([0, 2, 7]) # ndarray
        ind_diag=diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=dot(transpose(self.L[:,asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(allclose(X, 0.0))

        k=5 # number of eigenvalues
        ind=arange(k)
        ind_diag=diag_indices(len(ind))        
        Rn=decomposition.eigenvectors(self.A, k=ind)
        X=dot(transpose(self.L[:,asarray(ind)]), Rn)
        X[ind_diag]=0.0    
        self.assertTrue(allclose(X, 0.0))

    def test_rdl_decomposition(self):
        vn, Ln, Rn=decomposition.rdl_decomposition(self.A)

        """Eigenvalues"""
        self.assertTrue(allclose(self.v, vn))

        """Eigenvectors"""
        ind_diag=diag_indices(self.dim)
        
        X=dot(transpose(self.L), Rn)
        X[ind_diag]=0.0
        self.assertTrue(allclose(X, 0.0))

        # X=dot(transpose(Ln), self.R)
        # X[ind_diag]=0.0
        # self.assertTrue(allclose(X, 0.0))

        """
        The computed left eigenvectors Ln are sometimes 
        not perfectly orthogonal to the true right eienvectors
        R. This is why the above test does sometimes fail.
        """
        
        """They are however 'good' left eigenvectors for A"""
        X=dot(transpose(self.A), Ln)
        self.assertTrue(allclose(X, vn[newaxis, :]*Ln))

    def test_timescales(self):
        ts_n=decomposition.timescales(self.A)
        ts=zeros(len(self.v))
        ts[0]=inf
        ts[1:]=-1.0/log(self.v[1:])
        self.assertTrue(allclose(ts, ts_n))

        v=random.rand(self.dim)
        v[0:5]=1.0
        
class TestTimescales(unittest.TestCase):

    def setUp(self):
        self.dim=100

        """Random left eigenvectors"""
        self.L=random_linearly_independent_vectors(self.dim, self.dim)
        """Corresponding right eigenvectors"""
        self.R=solve(transpose(self.L), eye(self.dim))    

        """Purely real spectrum, unique eigenvalue with modulus one"""
        v_real=linspace(0.1, 1.0, self.dim)
        ind=argsort(abs(v_real))[::-1]
        self.v_real=v_real[ind]
        self.A_real=dot(self.R, dot(diag(self.v_real), transpose(self.L)))
        self.ts_real=zeros(len(self.v_real))
        self.ts_real[0]=inf
        self.ts_real[1:]=-1.0/log(abs(self.v_real[1:]))

        """Complex spectrum, unique eigenvalue with modulus one"""
        v_complex=linspace(0.1, 1.0, self.dim)+0.0*1j
        v_complex[1:5]=0.9+0.1*1j
        ind=argsort(abs(v_complex))[::-1]
        self.v_complex=v_complex[ind]
        self.A_complex=dot(self.R, dot(diag(self.v_complex), transpose(self.L)))
        self.ts_complex=zeros(len(self.v_complex))
        self.ts_complex[0]=inf
        self.ts_complex[1:]=-1.0/log(abs(self.v_complex[1:]))

        """Purely real spectrum, multiple eigenvalues with modulus one"""
        v_real_m=linspace(0.1, 1.0, self.dim)
        ind=argsort(abs(v_real_m))[::-1]
        self.v_real_m=v_real_m[ind]
        self.v_real_m[1:5]=1.0
        self.A_real_m=dot(self.R, dot(diag(self.v_real_m), transpose(self.L)))
        self.ts_real_m=zeros(len(self.v_real_m))
        self.ts_real_m[0:5]=inf
        self.ts_real_m[5:]=-1.0/log(abs(self.v_real_m[5:]))

        """Complex spectrum, multiple eigenvalues with modulus one"""
        v_complex_m=linspace(0.1, 1.0, self.dim)+0.0*1j
        ind=argsort(abs(v_complex_m))[::-1]
        self.v_complex_m=v_complex_m[ind]
        self.v_complex_m[1:5]=(1.0+1.0*1j)/sqrt(2.0)
        self.A_complex_m=dot(self.R, dot(diag(self.v_complex_m), transpose(self.L)))
        self.ts_complex_m=zeros(len(self.v_complex_m))
        self.ts_complex_m[0:5]=inf
        self.ts_complex_m[5:]=-1.0/log(abs(self.v_complex_m[5:]))      

       
    def tearDown(self):
        pass

    def mdot(self, *args):
        return reduce(numpy.dot, args)
    
    def test_timescales(self):
        """tau=1"""
        ts_n=decomposition.timescales(self.A_real)
        self.assertTrue(allclose(ts_n, self.ts_real))
        
        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_complex)
            self.assertTrue(allclose(ts_n, self.ts_complex))

        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_real_m)
            self.assertTrue(allclose(ts_n, self.ts_real_m))
        
        with self.assertRaises(RuntimeWarning):
            ts_n=decomposition.timescales(self.A_complex_m)
            self.assertTrue(allclose(ts_n, self.ts_complex_m))           

        """tau=10"""
        ts_n=decomposition.timescales(self.A_real, tau=10)
        self.assertTrue(allclose(ts_n, 10*self.ts_real))

        """tau=10, k=8"""
        ts_n=decomposition.timescales(self.A_real, tau=10, k=8)
        self.assertTrue(allclose(ts_n, 10*self.ts_real[0:8]))
        
        
if __name__=="__main__":
    unittest.main()
        
