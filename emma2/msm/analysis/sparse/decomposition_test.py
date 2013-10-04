"""Test package for the decomposition module"""
import unittest

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse

import decomposition

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

def random_orthonormal_vectors(d, k):
    r"""Generate a random set of k-orthonormal vectors

    The algorithm picks k random points uniformly distributed
    on the d-sphere. They will form a set of linear independent
    vectors with probability one for k<=d.

    A subsequent QR-decomposition produces an orthonormal set"""
    X=random_linearly_independent_vectors(d, k)
    Q, R=scipy.linalg.qr(X, mode='economic')
    return Q

class TestDecomposition(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mu(self):
        """Generate a random kxk dense transition matrix"""
        k=100
        C=np.random.random_integers(0, 100, size=(k, k))
        T=1.0*C/np.sum(C, axis=1)[:, np.newaxis]
        v, L, R=scipy.linalg.eig(T, left=True, right=True)
        nu=L[:,0]
        statdist=nu/np.sum(nu)
        
        """Convert to sparse matrix"""
        T_sparse=scipy.sparse.csr_matrix(T)
        statdist_sparse=decomposition.mu(T_sparse)

        self.assertTrue(np.allclose(statdist, statdist_sparse))

if __name__=="__main__":
    unittest.main()

    
    
    
    
    
    

