
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

from pyemma.util.numeric import assert_allclose
import scipy.sparse

import transition_matrix

"""Unit tests for the transition_matrix module"""


class TestTransitionMatrixNonReversible(unittest.TestCase):
    def setUp(self):
        """Small test cases"""
        self.C1 = scipy.sparse.csr_matrix([[1, 3], [3, 1]])
        self.C2 = scipy.sparse.csr_matrix([[0, 2], [1, 1]])

        self.T1 = scipy.sparse.csr_matrix([[0.25, 0.75], [0.75, 0.25]])
        self.T2 = scipy.sparse.csr_matrix([[0, 1], [0.5, 0.5]])

        """Zero row sum throws an error"""
        self.C0 = scipy.sparse.csr_matrix([[0, 0], [3, 1]])

    def tearDown(self):
        pass

    def test_count_matrix(self):
        """Small test cases"""
        T = transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        assert_allclose(T, self.T1.toarray())

        T = transition_matrix.transition_matrix_non_reversible(self.C1).toarray()
        assert_allclose(T, self.T1.toarray())


if __name__ == "__main__":
    unittest.main()