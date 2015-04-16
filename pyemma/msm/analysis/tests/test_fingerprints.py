
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

r"""Unit test for the fingerprint API-functions

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from pyemma.msm.analysis import rdl_decomposition, timescales
from pyemma.msm.analysis import fingerprint_correlation, fingerprint_relaxation
from pyemma.msm.analysis import expectation, correlation, relaxation

from birth_death_chain import BirthDeathChain

################################################################################
# Dense
################################################################################


class TestFingerprintDense(unittest.TestCase):
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
        R, D, L = rdl_decomposition(self.T)
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4
        self.tau = 7.5

        """Observables"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2

        """Initial vector for relaxation"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_fingerprint_correlation(self):
        """Autocorrelation"""

        """k=None, tau=1"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1)
        assert_allclose(tsn, self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=None, tau=7.5"""
        tau = self.tau
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=4, tau=1"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(acorr_ampn, acorr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts[0:k])
        assert_allclose(acorr_ampn, acorr_amp)

        """Cross-correlation"""

        """k=None, tau=1"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2)
        assert_allclose(tsn, self.ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=None, tau=7.5"""
        tau = self.tau
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=4, tau=1"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(corr_ampn, corr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts[0:k])
        assert_allclose(corr_ampn, corr_amp)

    def test_fingerprint_relaxation(self):
        one_vec = np.ones(self.T.shape[0])

        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs1)
        tsn, relax_ampn = fingerprint_relaxation(self.T, self.p0, self.obs1)
        assert_allclose(tsn, self.ts)
        assert_allclose(relax_ampn, relax_amp)

        """k=4"""
        k = self.k
        relax_amp = np.dot(self.p0, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        tsn, relax_ampn = fingerprint_relaxation(self.T, self.p0, self.obs1, k=k)
        assert_allclose(tsn, self.ts[0:k])
        assert_allclose(relax_ampn, relax_amp)


class TestExpectation(unittest.TestCase):
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

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1

        self.obs1 = obs1

    def test_expectation(self):
        exp = np.dot(self.mu, self.obs1)
        expn = expectation(self.T, self.obs1)
        assert_allclose(exp, expn)


class TestCorrelationDense(unittest.TestCase):
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
        R, D, L = rdl_decomposition(self.T, norm='reversible')
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2
        self.one_vec = np.ones(10)

    def test_correlation(self):
        """Auto-correlation"""

        """k=None"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        acorr = np.dot(self.ev_t, acorr_amp)
        acorrn = correlation(self.T, self.obs1, times=self.times)
        assert_allclose(acorrn, acorr)

        """k=4"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs1)
        acorr = np.dot(self.ev_t[:, 0:k], acorr_amp)
        acorrn = correlation(self.T, self.obs1, times=self.times, k=k)
        assert_allclose(acorrn, acorr)

        """Cross-correlation"""

        """k=None"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        corr = np.dot(self.ev_t, corr_amp)
        corrn = correlation(self.T, self.obs1, obs2=self.obs2, times=self.times)
        assert_allclose(corrn, corr)

        """k=4"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs2)
        corr = np.dot(self.ev_t[:, 0:k], corr_amp)
        corrn = correlation(self.T, self.obs1, obs2=self.obs2, times=self.times, k=k)
        assert_allclose(corrn, corr)


class TestRelaxationDense(unittest.TestCase):
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

        """Test matrix-vector product against spectral decomposition"""
        R, D, L = rdl_decomposition(self.T)
        self.L = L
        self.R = R
        self.ts = timescales(self.T)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.k = 4

        """Observable"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        self.obs = obs1

        """Initial distribution"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_relaxation(self):
        """k=None"""
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs)
        relax = np.dot(self.ev_t, relax_amp)
        relaxn = relaxation(self.T, self.p0, self.obs, times=self.times)
        assert_allclose(relaxn, relax)

        """k=4"""
        k = self.k
        relax_amp = np.dot(self.p0, self.R[:, 0:k]) * np.dot(self.L[0:k, :], self.obs)
        relax = np.dot(self.ev_t[:, 0:k], relax_amp)
        relaxn = relaxation(self.T, self.p0, self.obs, k=k, times=self.times)
        assert_allclose(relaxn, relax)


# ===================
# Sparse
# ===================

class TestFingerprintSparse(unittest.TestCase):
    def setUp(self):
        self.k = 4

        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)

        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix_sparse()
        R, D, L = rdl_decomposition(self.T, k=self.k)
        self.L = L
        self.R = R
        self.ts = timescales(self.T, k=self.k)
        self.times = np.array([1, 5, 10, 20])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        self.tau = 7.5

        """Observables"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2

        """Initial vector for relaxation"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_fingerprint_correlation(self):
        """Autocorrelation"""

        """k=4, tau=1"""
        k = self.k
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k)
        assert_allclose(tsn, self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, acorr_ampn = fingerprint_correlation(self.T, self.obs1, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(acorr_ampn, acorr_amp)

        """Cross-correlation"""

        """k=4, tau=1"""
        k = self.k
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k)
        assert_allclose(tsn, self.ts)
        assert_allclose(corr_ampn, corr_amp)

        """k=4, tau=7.5"""
        tau = self.tau
        tsn, corr_ampn = fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=k, tau=tau)
        assert_allclose(tsn, tau * self.ts)
        assert_allclose(corr_ampn, corr_amp)

    def test_fingerprint_relaxation(self):
        one_vec = np.ones(self.T.shape[0])

        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs1)
        tsn, relax_ampn = fingerprint_relaxation(self.T, self.p0, self.obs1, k=self.k)
        assert_allclose(tsn, self.ts)
        assert_allclose(relax_ampn, relax_amp)


class TestCorrelationSparse(unittest.TestCase):
    def setUp(self):
        self.k = 4

        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)

        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix_sparse()
        R, D, L = rdl_decomposition(self.T, k=self.k)
        self.L = L
        self.R = R
        self.ts = timescales(self.T, k=self.k)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1 = obs1
        self.obs2 = obs2
        self.one_vec = np.ones(10)

    def test_correlation(self):
        """Auto-correlation"""
        acorr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs1)
        acorr = np.dot(self.ev_t, acorr_amp)
        acorrn = correlation(self.T, self.obs1, k=self.k, times=self.times)
        assert_allclose(acorrn, acorr)

        """Cross-correlation"""
        corr_amp = np.dot(self.mu * self.obs1, self.R) * np.dot(self.L, self.obs2)
        corr = np.dot(self.ev_t, corr_amp)
        corrn = correlation(self.T, self.obs1, obs2=self.obs2, k=self.k, times=self.times)
        assert_allclose(corrn, corr)


class TestRelaxationSparse(unittest.TestCase):
    def setUp(self):
        self.k = 4

        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)

        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix_sparse()

        """Test matrix-vector product against spectral decomposition"""
        R, D, L = rdl_decomposition(self.T, k=self.k)
        self.L = L
        self.R = R
        self.ts = timescales(self.T, k=self.k)
        self.times = np.array([1, 5, 10, 20, 100])

        ev = np.diagonal(D)
        self.ev_t = ev[np.newaxis, :] ** self.times[:, np.newaxis]

        """Observable"""
        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        self.obs = obs1

        """Initial distribution"""
        w0 = np.zeros(10)
        w0[0:4] = 0.25
        self.p0 = w0

    def test_relaxation(self):
        relax_amp = np.dot(self.p0, self.R) * np.dot(self.L, self.obs)
        relax = np.dot(self.ev_t, relax_amp)
        relaxn = relaxation(self.T, self.p0, self.obs, k=self.k, times=self.times)
        assert_allclose(relaxn, relax)


if __name__ == "__main__":
    unittest.main()