import unittest
import numpy as np
import scipy
import scipy.sparse
from pyemma.msm.estimation.sparse.mle_trev import mle_trev as mtrs
from pyemma.msm.estimation.dense.transition_matrix import estimate_transition_matrix_reversible as etmr
from pyemma.msm.estimation import is_connected
from pyemma.msm.estimation import tmatrix as apicall

class Test_mle_trev(unittest.TestCase):
    def test_mle_trev(self):
        C=np.loadtxt('testfiles/C_1_lag.dat')
        
        T_cython_sparse = mtrs(scipy.sparse.csr_matrix(C)).toarray()
            
        T_python = etmr(C)
        T_api_dense = apicall(C,reversible=True)
        T_api_sparse = apicall(scipy.sparse.csr_matrix(C),reversible=True).toarray() 

        self.assertTrue(np.allclose(T_cython_sparse,T_python))
        self.assertTrue(np.allclose(T_api_sparse,T_python))
        self.assertTrue(np.allclose(T_api_dense,T_python))            


if __name__ == '__main__':
    unittest.main()
    
