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
