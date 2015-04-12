import unittest
import numpy as np
from pyemma.util.numeric import assert_allclose
import scipy
import scipy.sparse

from os.path import abspath, join
from os import pardir

from pyemma.msm.estimation.sparse.mle_trev import mle_trev as mtrs
from pyemma.msm.estimation.dense.transition_matrix import estimate_transition_matrix_reversible as etmr
from pyemma.msm.estimation import tmatrix as apicall

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class Test_mle_trev(unittest.TestCase):
    def test_mle_trev(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')

        T_cython_sparse = mtrs(scipy.sparse.csr_matrix(C)).toarray()

        T_python = etmr(C)
        T_api_dense = apicall(C, reversible=True)
        T_api_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True).toarray()

        assert_allclose(T_cython_sparse, T_python)
        assert_allclose(T_api_sparse, T_python)
        assert_allclose(T_api_dense, T_python)


if __name__ == '__main__':
    unittest.main()
