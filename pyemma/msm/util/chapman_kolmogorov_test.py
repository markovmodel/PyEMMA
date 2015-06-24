
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

r"""Unit test for Chapman-Kolmogorov-Test module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np

from pyemma.msm.generation import generate_traj
from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist
from pyemma.util.numeric import assert_allclose

from birth_death_chain import BirthDeathChain
from chapman_kolmogorov import cktest


class TestCkTestBirthDeath(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(42)

        """Meta-stable birth-death chain"""
        b = 2
        q = np.zeros(7)
        p = np.zeros(7)
        q[1:] = 0.5
        p[0:-1] = 0.5
        q[2] = 1.0 - 10 ** (-b)
        q[4] = 10 ** (-b)
        p[2] = 10 ** (-b)
        p[4] = 1.0 - 10 ** (-b)

        bdc = BirthDeathChain(q, p)
        P = bdc.transition_matrix()
        dtraj = generate_traj(P, 10000, start=0)
        tau = 1

        """Estimate MSM"""
        C_MSM = cmatrix(dtraj, tau)
        lcc_MSM = largest_connected_set(C_MSM)
        Ccc_MSM = connected_cmatrix(C_MSM, lcc=lcc_MSM)
        P_MSM = tmatrix(Ccc_MSM)
        mu_MSM = statdist(P_MSM)

        """Meta-stable sets"""
        A = [0, 1, 2]
        B = [4, 5, 6]

        w_MSM = np.zeros((2, mu_MSM.shape[0]))
        w_MSM[0, A] = mu_MSM[A] / mu_MSM[A].sum()
        w_MSM[1, B] = mu_MSM[B] / mu_MSM[B].sum()

        K = 10
        P_MSM_dense = P_MSM.toarray()

        p_MSM = np.zeros((K, 2))
        w_MSM_k = 1.0 * w_MSM
        for k in range(1, K):
            w_MSM_k = np.dot(w_MSM_k, P_MSM_dense)
            p_MSM[k, 0] = w_MSM_k[0, A].sum()
            p_MSM[k, 1] = w_MSM_k[1, B].sum()

        """Assume that sets are equal, A(\tau)=A(k \tau) for all k"""
        w_MD = 1.0 * w_MSM
        p_MD = np.zeros((K, 2))
        eps_MD = np.zeros((K, 2))
        p_MSM[0, :] = 1.0
        p_MD[0, :] = 1.0
        eps_MD[0, :] = 0.0
        for k in range(1, K):
            """Build MSM at lagtime k*tau"""
            C_MD = cmatrix(dtraj, k * tau, sliding=True) / (k * tau)
            lcc_MD = largest_connected_set(C_MD)
            Ccc_MD = connected_cmatrix(C_MD, lcc=lcc_MD)
            c_MD = Ccc_MD.sum(axis=1)
            P_MD = tmatrix(Ccc_MD).toarray()
            w_MD_k = np.dot(w_MD, P_MD)

            """Set A"""
            prob_MD = w_MD_k[0, A].sum()
            c = c_MD[A].sum()
            p_MD[k, 0] = prob_MD
            eps_MD[k, 0] = np.sqrt(k * (prob_MD - prob_MD ** 2) / c)

            """Set B"""
            prob_MD = w_MD_k[1, B].sum()
            c = c_MD[B].sum()
            p_MD[k, 1] = prob_MD
            eps_MD[k, 1] = np.sqrt(k * (prob_MD - prob_MD ** 2) / c)

        """Input"""
        self.P_MSM = P_MSM
        self.lcc_MSM = lcc_MSM
        self.dtraj = dtraj
        self.tau = tau
        self.K = K
        self.A = A
        self.B = B

        """Expected results"""
        self.p_MSM = p_MSM
        self.p_MD = p_MD
        self.eps_MD = eps_MD

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_cktest(self):
        p_MSM, p_MD, eps_MD = cktest(self.P_MSM, self.lcc_MSM, self.dtraj,
                                     self.tau, self.K, sets=[self.A, self.B])

        assert_allclose(p_MSM, self.p_MSM)
        assert_allclose(p_MD, self.p_MD)
        assert_allclose(eps_MD, self.eps_MD)


if __name__ == "__main__":
    unittest.main()