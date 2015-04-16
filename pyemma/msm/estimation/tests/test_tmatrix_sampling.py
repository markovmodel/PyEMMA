
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

__author__ = 'noe'
import unittest
import warnings

import numpy as np
from pyemma.util.numeric import assert_allclose
import scipy.sparse

from pyemma.msm.estimation import sample_tmatrix, tmatrix_sampler
from pyemma.msm.analysis import is_transition_matrix

"""Unit tests for the transition_matrix module"""


class TestTransitionMatrixSampling(unittest.TestCase):
    def setUp(self):
        self.C = np.array([[7,1],
                           [2,2]])

    def test_sample_nonrev_1(self):
        P = sample_tmatrix(self.C)
        assert np.all(P.shape == self.C.shape)
        assert is_transition_matrix(P)

        # same with boject
        sampler = tmatrix_sampler(self.C)
        P = sampler.sample()
        assert np.all(P.shape == self.C.shape)
        assert is_transition_matrix(P)

    def test_sample_nonrev_10(self):
        sampler = tmatrix_sampler(self.C)
        Ps = sampler.sample(nsample = 10)
        assert len(Ps) == 10
        for i in range(10):
            assert np.all(Ps[i].shape == self.C.shape)
            assert is_transition_matrix(Ps[i])



if __name__ == "__main__":
    unittest.main()