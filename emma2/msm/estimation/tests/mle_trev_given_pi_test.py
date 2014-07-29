import unittest
import numpy as np
from emma2.msm.estimation.dense.mle_trev_given_pi import mle_trev_given_pi as mtrgpd
from emma2.msm.estimation.sparse.mle_trev_given_pi import mle_trev_given_pi as mtrgps
from emma2.msm.estimation.dense.transition_matrix import transition_matrix_reversible_fixpi as tmrfp

class Test_mle_trev_given_pi(unittest.TestCase):
    
    def setUp(self):
        self.eps = 1.0E-6
    
    def test_mle_trev_given_pi(self):
        for n in xrange(3,100):
            C  = (1000*np.random.rand(n,n)).astype(int).astype(float)
            pi = np.random.rand(n)
            pi /= sum(pi)   

            T_cython_dense = mtrgpd(C,pi,eps=self.eps)
            T_cython_sparse = mtrgps(C,pi,eps=self.eps)

            for i in xrange(n):
                if C[i,i] == 0:
                    C[i,i] = self.eps
            T_python = tmrfp(C,pi)

            self.assertTrue(np.allclose(T_cython_dense,T_python))
            self.assertTrue(np.allclose(T_cython_sparse,T_python))

if __name__ == '__main__':
    unittest.main()
    
