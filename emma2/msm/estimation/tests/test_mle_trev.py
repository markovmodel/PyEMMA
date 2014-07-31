import unittest
import numpy as np
import scipy
import scipy.sparse
#from emma2.msm.estimation.sparse.mle_trev import mle_trev as mtrs
#from emma2.msm.estimation.dense.transition_matrix import estimate_transition_matrix_reversible as etmr
from emma2.msm.estimation import tmatrix as apicall

class Test_mle_trev(unittest.TestCase):
    def test_mle_trev(self):
        for n in xrange(3,100):
            C  = (1000*np.random.rand(n,n)).astype(int)
            #x = np.random.rand(n)

            #print '0'
            #T_cython_sparse = mtrs(scipy.sparse.csr_matrix(C)).toarray()

            
            #print '1'
            #T_python = etmr(C)
            #print '2'
            T_api_dense = apicall(C,reversible=True)
            #print '3'
            #T_api_sparse = apicall(scipy.sparse.csr_matrix(C),reversible=True,Xinit=x).toarray()
            #print '4'

            #self.assertTrue(np.allclose(T_cython_dense,T_python))
            #self.assertTrue(np.allclose(T_api_sparse,T_python))
            #self.assertTrue(np.allclose(T_api_dense,T_python))

if __name__ == '__main__':
    unittest.main()
    
