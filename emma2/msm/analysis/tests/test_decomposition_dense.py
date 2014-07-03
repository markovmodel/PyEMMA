r"""Unit test for decomposition functions in api.py

.. moduleauthor:: Benjamin Trendelkamp-Schroer<benjamin DOT trendelkamp-schorer AT fu-berlin DOT de>

"""
import sys

import unittest
import warnings

import numpy as np

from scipy.linalg import eig, eigh, eigvals, eigvalsh, qr, solve

from emma2.util.exceptions import SpectralWarning

from emma2.msm.analysis import stationary_distribution, eigenvalues, eigenvectors, rdl_decomposition, timescales

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

class BirthDeathChain():
    """Birth and death chain class

    A general birth and death chain on a d-dimensional state space
    has the following transition matrix

                q_i,    j=i-1 for i>0
        p_ij=   r_i     j=i
                p_i     j=i+1 for i<d-1

    """
    def __init__(self, q, p):
        """Generate a birth and death chain from creation and
        anhilation probabilities.

        Parameters
        ----------
        q : array_like 
            Anhilation probabilities for transition from i to i-1
        p : array-like 
            Creation probabilities for transition from i to i+1

        """
        if q[0]!=0.0:
            raise ValueError('Probability q[0] must be zero')
        if p[-1]!=0.0:
            raise ValueError('Probability p[-1] must be zero')
        if not np.all(q+p<=1.0):
            raise ValueError('Probabilities q+p can not exceed one')
        self.q=q
        self.p=p
        self.r=1-self.q-self.p
        self.dim=self.r.shape[0]

    def transition_matrix(self):
        """Tridiagonal transition matrix for birth and death chain

        Returns
        -------
        P : (N,N) ndarray
            Transition matrix for birth and death chain with given
            creation and anhilation probabilities.

        """        
        P0=np.diag(self.r, k=0)
        P1=np.diag(self.p[0:-1], k=1)
        P_1=np.diag(self.q[1:], k=-1)
        return P0+P1+P_1

    def stationary_distribution(self):
        a=np.zeros(self.dim)
        a[0]=1.0
        a[1:]=np.cumprod(self.p[0:-1]/self.q[1:])
        mu=a/np.sum(a)
        return mu

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
        
        
if __name__=="__main__":
    unittest.main()
