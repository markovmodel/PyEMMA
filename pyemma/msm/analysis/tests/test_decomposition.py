r"""Unit test for decomposition functions in api.py

.. moduleauthor:: Benjamin Trendelkamp-Schroer<benjamin DOT trendelkamp-schorer AT fu-berlin DOT de>

"""
import unittest
import warnings

import numpy as np

from scipy.linalg import eig, eigvals
from scipy.sparse import csr_matrix

from pyemma.util.exceptions import SpectralWarning, ImaginaryEigenValueWarning
from birth_death_chain import BirthDeathChain

from pyemma.msm.analysis import stationary_distribution, eigenvalues, eigenvectors
from pyemma.msm.analysis import rdl_decomposition, timescales

################################################################################
# Dense
################################################################################

class TestDecompositionDense(unittest.TestCase):
    def setUp(self):
        self.dim=100
        self.k=10
        
        """Set up meta-stable birth-death chain"""
        p=np.zeros(self.dim)
        p[0:-1]=0.5
        
        q=np.zeros(self.dim)
        q[1:]=0.5

        p[self.dim/2-1]=0.001
        q[self.dim/2+1]=0.001
        
        self.bdc=BirthDeathChain(q, p)

    def test_statdist(self):
        P=self.bdc.transition_matrix()
        mu=self.bdc.stationary_distribution()
        mun=stationary_distribution(P)
        self.assertTrue(np.allclose(mu, mun))

    def test_eigenvalues(self):
        P=self.bdc.transition_matrix()
        ev=eigvals(P)
        """Sort with decreasing magnitude"""
        ev=ev[np.argsort(np.abs(ev))[::-1]]
        
        """k=None"""
        evn=eigenvalues(P)
        self.assertTrue(np.allclose(ev, evn))
        
        """k is not None"""
        evn=eigenvalues(P, k=self.k)
        self.assertTrue(np.allclose(ev[0:self.k], evn))

    def test_eigenvectors(self):
        P=self.bdc.transition_matrix()
        ev, L, R=eig(P, left=True, right=True)
        ind=np.argsort(np.abs(ev))[::-1]
        R=R[:,ind]
        L=L[:,ind]        

        """k=None"""
        Rn=eigenvectors(P)
        self.assertTrue(np.allclose(R, Rn))

        Ln=eigenvectors(P, right=False)
        self.assertTrue(np.allclose(L, Ln))

        """k is not None"""
        Rn=eigenvectors(P, k=self.k)
        self.assertTrue(np.allclose(R[:,0:self.k], Rn))

        Ln=eigenvectors(P, right=False, k=self.k)
        self.assertTrue(np.allclose(L[:,0:self.k], Ln))

    def test_rdl_decomposition(self):
        P=self.bdc.transition_matrix()
        mu=self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        Rn, Dn, Ln=rdl_decomposition(P)        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(P, Rn), np.dot(Rn, Dn)))
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(Ln, P), np.dot(Dn, Ln)))
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.dim)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))

        """k is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, k=self.k)        
        Xn=np.dot(Ln, Rn)               
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(P, Rn), np.dot(Rn, Dn)))
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(Ln, P), np.dot(Dn, Ln)))
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))

        """Reversible"""

        """k=None"""
        Rn, Dn, Ln=rdl_decomposition(P, norm='reversible')        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(P, Rn), np.dot(Rn, Dn)))
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(Ln, P), np.dot(Dn, Ln)))
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.dim)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))   
        """Reversibility"""
        self.assertTrue(np.allclose(Ln.transpose(), mu[:,np.newaxis]*Rn))

        """k is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, norm='reversible', k=self.k)        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(P, Rn), np.dot(Rn, Dn)))
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(np.dot(Ln, P), np.dot(Dn, Ln)))
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))   
        """Reversibility"""
        self.assertTrue(np.allclose(Ln.transpose(), mu[:,np.newaxis]*Rn))

    def test_timescales(self):
        P=self.bdc.transition_matrix()
        ev=eigvals(P)
        """Sort with decreasing magnitude"""
        ev=ev[np.argsort(np.abs(ev))[::-1]]
        ts=-1.0/np.log(np.abs(ev))

        """k=None"""
        tsn=timescales(P)
        self.assertTrue(np.allclose(ts[1:], tsn[1:]))

        """k is not None"""
        tsn=timescales(P, k=self.k)
        self.assertTrue(np.allclose(ts[1:self.k], tsn[1:]))
        

        """tau=7"""
        
        """k=None"""
        tsn=timescales(P, tau=7)
        self.assertTrue(np.allclose(7*ts[1:], tsn[1:]))

        """k is not None"""
        tsn=timescales(P, k=self.k, tau=7)
        self.assertTrue(np.allclose(7*ts[1:self.k], tsn[1:]))

class TestTimescalesDense(unittest.TestCase):
    
    def setUp(self):
        self.T=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self.P=np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
        self.W=np.array([[0, 1], [1, 0]])

    def test_timescales_1(self):
        """Multiple eigenvalues of magnitude one,
        eigenvalues with non-zero imaginary part"""
        ts=np.array([np.inf, np.inf])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsn=timescales(self.W)
            self.assertTrue(np.allclose(tsn, ts))
            assert issubclass(w[-1].category, SpectralWarning)

    def test_timescales_2(self):
        """Eigenvalues with non-zero imaginary part"""
        ts=np.array([np.inf,  0.971044,  0.971044])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsn=timescales(0.5*self.T+0.5*self.P)
            self.assertTrue(np.allclose(tsn, ts))
            assert issubclass(w[-1].category, ImaginaryEigenValueWarning)        

################################################################################
# Sparse
################################################################################

class TestDecompositionSparse(unittest.TestCase):
    def setUp(self):
        self.dim=100
        self.k=10
        self.ncv=40
        
        """Set up meta-stable birth-death chain"""
        p=np.zeros(self.dim)
        p[0:-1]=0.5
        
        q=np.zeros(self.dim)
        q[1:]=0.5

        p[self.dim/2-1]=0.001
        q[self.dim/2+1]=0.001
        
        self.bdc=BirthDeathChain(q, p)

    def test_statdist(self):
        P=self.bdc.transition_matrix_sparse()
        mu=self.bdc.stationary_distribution()
        mun=stationary_distribution(P)
        self.assertTrue(np.allclose(mu, mun))

    def test_eigenvalues(self):
        P=self.bdc.transition_matrix_sparse()
        P_dense=self.bdc.transition_matrix()
        ev=eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev=ev[np.argsort(np.abs(ev))[::-1]]
        
        """k=None"""
        with self.assertRaises(ValueError):
            evn=eigenvalues(P)
        
        """k is not None"""
        evn=eigenvalues(P, k=self.k)
        self.assertTrue(np.allclose(ev[0:self.k], evn))

        """k is not None and ncv is not None"""
        evn=eigenvalues(P, k=self.k, ncv=self.ncv)
        self.assertTrue(np.allclose(ev[0:self.k], evn))

    def test_eigenvectors(self):
        P_dense=self.bdc.transition_matrix()
        P=self.bdc.transition_matrix_sparse()
        ev, L, R=eig(P_dense, left=True, right=True)
        ind=np.argsort(np.abs(ev))[::-1]
        ev=ev[ind]
        R=R[:,ind]
        L=L[:,ind]        
        vals=ev[0:self.k]

        """k=None"""
        with self.assertRaises(ValueError):
            Rn=eigenvectors(P)

        with self.assertRaises(ValueError):
            Ln=eigenvectors(P, right=False)

        """k is not None"""
        Rn=eigenvectors(P, k=self.k)        
        self.assertTrue(np.allclose(vals[np.newaxis,:]*Rn, P.dot(Rn)))

        Ln=eigenvectors(P, right=False, k=self.k)
        self.assertTrue(np.allclose(P.transpose().dot(Ln), vals[np.newaxis,:]*Ln))

        """k is not None and ncv is not None"""
        Rn=eigenvectors(P, k=self.k, ncv=self.ncv)        
        self.assertTrue(np.allclose(vals[np.newaxis,:]*Rn, P.dot(Rn)))

        Ln=eigenvectors(P, right=False, k=self.k, ncv=self.ncv)
        self.assertTrue(np.allclose(P.transpose().dot(Ln), vals[np.newaxis,:]*Ln))

    def test_rdl_decomposition(self):
        P=self.bdc.transition_matrix_sparse()
        mu=self.bdc.stationary_distribution()

        """Non-reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln=rdl_decomposition(P)        

        """k is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, k=self.k)        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(P.dot(Rn), np.dot(Rn, Dn)))    
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln)))               
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))

        """k is not None, ncv is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, k=self.k, ncv=self.ncv)        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(P.dot(Rn), np.dot(Rn, Dn)))    
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln)))               
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))

        """Reversible"""

        """k=None"""
        with self.assertRaises(ValueError):
            Rn, Dn, Ln=rdl_decomposition(P, norm='reversible')        

        """k is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, k=self.k, norm='reversible')        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(P.dot(Rn), np.dot(Rn, Dn)))    
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln)))               
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))
        """Reversibility"""
        self.assertTrue(np.allclose(Ln.transpose(), mu[:,np.newaxis]*Rn))

        """k is not None ncv is not None"""
        Rn, Dn, Ln=rdl_decomposition(P, k=self.k, norm='reversible', ncv=self.ncv)        
        Xn=np.dot(Ln, Rn)
        """Right-eigenvectors"""
        self.assertTrue(np.allclose(P.dot(Rn), np.dot(Rn, Dn)))    
        """Left-eigenvectors"""
        self.assertTrue(np.allclose(P.transpose().dot(Ln.transpose()).transpose(), np.dot(Dn, Ln)))               
        """Orthonormality"""
        self.assertTrue(np.allclose(Xn, np.eye(self.k)))
        """Probability vector"""
        self.assertTrue(np.allclose(np.sum(Ln[0,:]), 1.0))
        """Reversibility"""
        self.assertTrue(np.allclose(Ln.transpose(), mu[:,np.newaxis]*Rn))

    def test_timescales(self):
        P_dense=self.bdc.transition_matrix()
        P=self.bdc.transition_matrix_sparse()
        ev=eigvals(P_dense)
        """Sort with decreasing magnitude"""
        ev=ev[np.argsort(np.abs(ev))[::-1]]
        ts=-1.0/np.log(np.abs(ev))

        """k=None"""
        with self.assertRaises(ValueError):
            tsn=timescales(P)

        """k is not None"""
        tsn=timescales(P, k=self.k)
        self.assertTrue(np.allclose(ts[1:self.k], tsn[1:]))

        """k is not None, ncv is not None"""
        tsn=timescales(P, k=self.k, ncv=self.ncv)
        self.assertTrue(np.allclose(ts[1:self.k], tsn[1:]))
        

        """tau=7"""      

        """k is not None"""
        tsn=timescales(P, k=self.k, tau=7)
        self.assertTrue(np.allclose(7*ts[1:self.k], tsn[1:]))
        
        
if __name__=="__main__":
    unittest.main()
