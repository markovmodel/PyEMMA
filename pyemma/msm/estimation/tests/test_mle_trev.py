
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
import numpy as np
from pyemma.util.numeric import assert_allclose
import scipy
import scipy.sparse
import warnings
import pyemma.util.exceptions

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

    def test_warnings(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mtrs(scipy.sparse.csr_matrix(C), maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, pyemma.util.exceptions.NotConvergedWarning)
            
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            etmr(C, maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, pyemma.util.exceptions.NotConvergedWarning)
            
    def test_noninteger_counts_sparse(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_sparse_reference = mtrs(scipy.sparse.csr_matrix(C)).toarray()
        T_sparse_scaled_1 = mtrs(scipy.sparse.csr_matrix(C*10.0)).toarray()
        T_sparse_scaled_2 = mtrs(scipy.sparse.csr_matrix(C*0.1)).toarray()
        assert_allclose(T_sparse_reference, T_sparse_scaled_1)
        assert_allclose(T_sparse_reference, T_sparse_scaled_2)

    def test_noninteger_counts_dense(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_dense_reference = etmr(C)
        T_dense_scaled_1 = etmr(C*10.0)
        T_dense_scaled_2 = etmr(C*0.1)
        assert_allclose(T_dense_reference, T_dense_scaled_1)
        assert_allclose(T_dense_reference, T_dense_scaled_2)


if __name__ == '__main__':
    unittest.main()
