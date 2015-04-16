
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

'''
Created on Jun 3, 2014

@author: marscher
'''
import unittest
import numpy as np
from pyemma.util.numeric import assert_allclose

from committor_test import BirthDeathChain

import correlations


class TestCorrelations(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)

        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix()

    def test_time_correlation(self):
        """
        since we have no overlap between observations and do not propagate the
        operator, the correlation is zero.
        P^0 = diag(1)
        """
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1
        time = 0
        corr = correlations.time_correlation_direct_by_mtx_vec_prod(self.T, self.mu, obs1, obs2, time)
        self.assertEqual(corr, 0)

        time = 100
        corr = correlations.time_correlation_direct_by_mtx_vec_prod(self.T, self.mu, obs1, obs2, time)
        self.assertGreater(corr, 0.0, "correlation should be > 0.")

    @unittest.SkipTest
    def test_time_auto_correlation(self):
        """test with obs2 = obs1, to test autocorrelation"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        time = 100
        print correlations.time_correlation_direct_by_mtx_vec_prod(self.T, self.mu, obs1, time=time)

    @unittest.SkipTest
    def test_time_corr2(self):
        obs1 = np.zeros(10)
        obs1[5:] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        time = 2
        print correlations.time_correlation_direct_by_mtx_vec_prod(self.T, self.mu, obs1, obs2, time=time)

    def test_time_correlations(self):
        """
        tests whether the outcome of the wrapper time_correlations_direct
        is equivalent to calls to time_correlation_direct_by_mtx_vec_prod with same time set.
        """
        obs1 = np.zeros(10)
        obs1[3:5] = 1
        obs2 = np.zeros(10)
        obs2[4:8] = 1
        times = [1, 2, 20, 40, 100, 200, 1000]
        # calculate without wrapper
        corr_expected = np.empty(len(times))
        i = 0
        for t in times:
            corr_expected[i] = correlations.time_correlation_direct_by_mtx_vec_prod(self.T, self.mu, obs1, obs2, t)
            i += 1
        # calculate with wrapper
        corr_actual = correlations.time_correlations_direct(self.T, self.mu, obs1, obs2, times)

        self.assertTrue(np.allclose(corr_expected, corr_actual),
                        "correlations differ:\n%s\n%s" % (corr_expected, corr_actual))

    def test_time_relaxation_stat(self):
        """
            start with stationary distribution, so increasing time should 
            not change relaxation any more.
        """
        obs = np.zeros(10)
        obs[9] = 1
        p0 = self.mu
        c1 = correlations.time_relaxation_direct_by_mtx_vec_prod(self.T, p0, obs, time=1)
        c1000 = correlations.time_relaxation_direct_by_mtx_vec_prod(self.T, p0, obs, time=1000)
        self.assertAlmostEqual(c1, c1000,
                               msg="relaxation should be same, since we start in equilibrium.")

    def test_time_relaxation(self):
        obs = np.zeros(10)
        obs[9] = 1

        p0 = np.zeros(10) * 1. / 10

        # compute by hand
        # p0 P^k obs
        P1000 = np.linalg.matrix_power(self.T, 1000)
        expected = np.dot(np.dot(p0, P1000), obs)
        result = correlations.time_relaxation_direct_by_mtx_vec_prod(self.T, p0, obs, time=1000)
        self.assertAlmostEqual(expected, result)

    def test_time_relaxations(self):
        obs = np.zeros(10)
        obs[9] = 1

        p0 = np.zeros(10) * 1. / 10
        times = [1, 100, 1000]
        expected = []
        for t in times:
            expected.append(correlations.time_relaxation_direct_by_mtx_vec_prod(self.T, p0, obs, t))

        result = correlations.time_relaxations_direct(self.T, p0, obs, times)

        assert_allclose(expected, result)


if __name__ == "__main__":
    unittest.main()