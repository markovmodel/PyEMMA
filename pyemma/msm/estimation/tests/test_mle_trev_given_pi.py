import unittest
import numpy as np
import scipy
import scipy.sparse
from pyemma.msm.estimation.dense.mle_trev_given_pi import mle_trev_given_pi as mtrgpd
from pyemma.msm.estimation.sparse.mle_trev_given_pi import mle_trev_given_pi as mtrgps
from pyemma.msm.estimation.dense.transition_matrix import transition_matrix_reversible_fixpi as tmrfp
from pyemma.msm.estimation import tmatrix as apicall

class Test_mle_trev_given_pi(unittest.TestCase):
    
    def setUp(self):
        self.eps = 1.0E-6
    
    def test_mle_trev_given_pi(self):
        C=np.loadtxt('testfiles/C_1_lag.dat')
        pi=np.loadtxt('testfiles/pi.dat')

        T_cython_dense = mtrgpd(C,pi,eps=self.eps)
        T_cython_sparse = mtrgps(scipy.sparse.csr_matrix(C),pi,eps=self.eps).toarray()
        T_python = tmrfp(C,pi)
        T_api_dense = apicall(C,reversible=True,mu=pi,eps=self.eps)
        T_api_sparse = apicall(scipy.sparse.csr_matrix(C),reversible=True,mu=pi,\
                                   eps=self.eps).toarray()

        # print T_python
        # print T_api_sparse

        self.assertTrue(np.allclose(T_cython_dense,T_python))
        self.assertTrue(np.allclose(T_cython_sparse,T_python))
        self.assertTrue(np.allclose(T_api_sparse,T_python))
        self.assertTrue(np.allclose(T_api_dense,T_python))

if __name__ == '__main__':
    unittest.main()
    
