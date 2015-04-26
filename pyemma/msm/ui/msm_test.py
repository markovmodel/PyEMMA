
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

r"""Unit test for the MSM module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np

from pyemma.msm.generation import generate_traj
from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist, timescales
from pyemma.util.numeric import assert_allclose

from birth_death_chain import BirthDeathChain

from pyemma.msm import estimate_markov_model


class TestMSM(unittest.TestCase):

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
        q[2] = 1.0-10**(-b)
        q[4] = 10**(-b)
        p[2] = 10**(-b)
        p[4] = 1.0-10**(-b)

        bdc = BirthDeathChain(q, p)
        P = bdc.transition_matrix()
        self.dtrajs = [generate_traj(P, 10000, start=0)]
        self.tau = 1

        """Estimate MSM"""
        self.C_MSM = cmatrix(self.dtrajs, self.tau, sliding=True).toarray()
        self.lcc_MSM = largest_connected_set(self.C_MSM)
        self.Ccc_MSM = connected_cmatrix(self.C_MSM, lcc=self.lcc_MSM)
        self.P_MSM = tmatrix(self.Ccc_MSM, reversible=True)
        self.mu_MSM = statdist(self.P_MSM)
        self.k = 3
        self.ts = timescales(self.P_MSM, k=self.k+1, tau=self.tau)[1:]

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_MSM(self):
        msm = estimate_markov_model(self.dtrajs, self.tau)
        assert_allclose(self.dtrajs, msm.discrete_trajectories_full)
        self.assertEqual(self.tau, msm.lagtime)
        assert_allclose(self.lcc_MSM, msm.largest_connected_set)
        assert_allclose(self.Ccc_MSM, msm.count_matrix_active)
        assert_allclose(self.C_MSM, msm.count_matrix_full)
        assert_allclose(self.P_MSM, msm.transition_matrix)
        assert_allclose(self.mu_MSM, msm.stationary_distribution)
        assert_allclose(self.ts, msm.timescales(self.k))

if __name__ == "__main__":
    unittest.main()