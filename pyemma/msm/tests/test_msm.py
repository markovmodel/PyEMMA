
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

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import unittest

import numpy as np
import warnings

from os.path import abspath, join
from os import pardir

from pyemma.msm.generation import generate_traj
from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist, timescales
from pyemma.util.numeric import assert_allclose
from pyemma.msm.ui.birth_death_chain import BirthDeathChain
from pyemma.msm import estimate_markov_model as markov_state_model


class TestMSMSimple(unittest.TestCase):
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
        self.dtraj = generate_traj(P, 10000, start=0)
        self.tau = 1

        """Estimate MSM"""
        self.C_MSM = cmatrix(self.dtraj, self.tau, sliding=True)
        self.lcc_MSM = largest_connected_set(self.C_MSM)
        self.Ccc_MSM = connected_cmatrix(self.C_MSM, lcc=self.lcc_MSM)
        self.P_MSM = tmatrix(self.Ccc_MSM, reversible=True)
        self.mu_MSM = statdist(self.P_MSM)
        self.k = 3
        self.ts = timescales(self.P_MSM, k=self.k, tau=self.tau)

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_MSM(self):
        msm = markov_state_model(self.dtraj, self.tau)
        assert_allclose(self.dtraj, msm.discrete_trajectories_full[0])
        self.assertEqual(self.tau, msm.lagtime)
        assert_allclose(self.lcc_MSM, msm.largest_connected_set)
        self.assertTrue(np.allclose(self.Ccc_MSM.toarray(), msm.count_matrix_active))
        self.assertTrue(np.allclose(self.C_MSM.toarray(), msm.count_matrix_full))
        self.assertTrue(np.allclose(self.P_MSM.toarray(), msm.transition_matrix))
        assert_allclose(self.mu_MSM, msm.stationary_distribution)
        assert_allclose(self.ts[1:], msm.timescales(self.k - 1))


class TestMSMDoubleWellReversible(unittest.TestCase):
    def setUp(self):
        testpath = abspath(join(abspath(__file__), pardir)) + '/../../util/tests/data/'
        import pyemma.util.discrete_trajectories as dt

        self.dtraj = dt.read_discrete_trajectory(testpath + '2well_traj_100K.dat')
        self.tau = 10
        self.msmrev = markov_state_model(self.dtraj, self.tau)
        self.msm = markov_state_model(self.dtraj, self.tau, reversible=False)

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def _compute(self, msm):
        # should give warning
        with warnings.catch_warnings(record=True) as w:
            msm.estimate()

    def test_compute(self):
        self._compute(self.msmrev)
        self._compute(self.msm)

    def _computed(self, msm):
        assert (msm.estimated)

    def test_computed(self):
        self._computed(self.msmrev)
        self._computed(self.msm)

    def test_reversible(self):
        # NONREVERSIBLE
        assert (self.msmrev.is_reversible)
        # REVERSIBLE
        assert (not self.msm.is_reversible)

    def _sparse(self, msm):
        assert (not msm.is_sparse)

    def test_sparse(self):
        self._sparse(self.msmrev)
        self._sparse(self.msm)

    def _lagtime(self, msm):
        assert (self.msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msm)

    def _active_set(self, msm):
        # should always be <= full set
        assert (len(msm.active_set) <= self.msm._n_full)
        # should be length of nstates
        assert (len(msm.active_set) == self.msm.nstates)

    def test_active_set(self):
        self._active_set(self.msmrev)
        self._active_set(self.msm)

    def _largest_connected_set(self, msm):
        lcs = msm.largest_connected_set
        # identical to first connected set
        assert (np.all(lcs == msm.connected_sets[0]))
        # LARGEST: identical to active set
        assert (np.all(lcs == msm.active_set))

    def test_largest_connected_set(self):
        self._largest_connected_set(self.msmrev)
        self._largest_connected_set(self.msm)

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm._n_full)
        # THIS DATASET:
        assert (msm.nstates == 66)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msm)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msm)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msm)

    def _count_matrix_active(self, msm):
        C = msm.count_matrix_active
        assert (np.all(C.shape == (msm.nstates, msm.nstates)))

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msm)

    def _count_matrix_full(self, msm):
        C = msm.count_matrix_full
        assert (np.all(C.shape == (msm._n_full, msm._n_full)))

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msm)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtraj == msm.discrete_trajectories_full[0]))

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msm)

    def _discrete_trajectories_active(self, msm):
        dta = msm.discrete_trajectories_active
        # HERE
        assert (len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert (dta[0][0] < self.dtraj[0])

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msm)

    def _timestep(self, msm):
        assert (msm.timestep.startswith('1'))
        assert (msm.timestep.endswith('step'))

    def test_timestep(self):
        self._timestep(self.msmrev)
        self._timestep(self.msm)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
        assert (isinstance(P, np.ndarray))
        # shape
        assert (np.all(P.shape == (msm.nstates, msm.nstates)))
        # test transition matrix properties
        import pyemma.msm.analysis as msmana

        assert (msmana.is_transition_matrix(P))
        assert (msmana.is_connected(P))
        # REVERSIBLE
        if msm.is_reversible:
            assert (msmana.is_reversible(P))

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_count_fraction(self, msm):
        # should always be a fraction
        assert (msm.active_count_fraction >= 0.0 and msm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (msm.active_count_fraction == 1.0)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.msmrev)
        self._active_count_fraction(self.msm)

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (msm.active_state_fraction >= 0.0 and msm.active_state_fraction <= 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msm)

    def _effective_count_matrix(self, msm):
        assert (np.allclose(self.tau * msm.effective_count_matrix, msm.count_matrix_active))

    def test_effective_count_matrix(self):
        self._effective_count_matrix(self.msmrev)
        self._effective_count_matrix(self.msm)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)

    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msm)

    def _eigenvalues(self, msm):
        ev = msm.eigenvalues()
        # stochasticity
        assert (np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        assert (np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            assert (evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(ev)))

    def test_eigenvalues(self):
        self._eigenvalues(self.msmrev)
        self._eigenvalues(self.msm)

    def _eigenvectors_left(self, msm):
        L = msm.eigenvectors_left()
        # shape should be right
        assert (np.all(L.shape == (msm.nstates, msm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0, :]
        err = msm.stationary_distribution - l1
        assert (np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert (np.allclose(np.sum(L[1:, :], axis=1), np.zeros(msm.nstates - 1)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(L)))

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.msmrev)
        self._eigenvectors_left(self.msm)

    def _eigenvectors_right(self, msm):
        R = msm.eigenvectors_right()
        # shape should be right
        assert (np.all(R.shape == (msm.nstates, msm.nstates)))
        # should be all ones
        r1 = R[:, 0]
        assert (np.allclose(r1, np.ones(msm.nstates)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(R)))

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.msmrev)
        self._eigenvectors_right(self.msm)

    def _eigenvectors_RDL(self, msm):
        R = msm.eigenvectors_right()
        D = np.diag(msm.eigenvalues())
        L = msm.eigenvectors_left()

        # orthogonality constraint
        assert (np.allclose(np.dot(R, L), np.eye(msm.nstates)))
        # REVERSIBLE: also true for LR because reversible matrix
        if msm.is_reversible:
            assert (np.allclose(np.dot(L, R), np.eye(msm.nstates)))
        # recover transition matrix
        assert (np.allclose(np.dot(R, np.dot(D, L)), msm.transition_matrix))

    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.msmrev)
        self._eigenvectors_RDL(self.msm)

    def _timescales(self, msm):
        ts = msm.timescales()
        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            assert (np.all(np.isreal(ts)))
            # HERE:
            assert (np.max(np.abs(ts[0:3] - np.array([310.87248087, 8.50933441, 5.09082957]))) < 1e-6)
        else:
            # HERE:
            assert (np.max(np.abs(ts[0:3] - np.array([310.49376926, 8.48302712, 5.02649564]))) < 1e-6)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, msm):
        a = 16
        b = 48
        q_forward = msm.committor_forward(a, b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)
        assert (np.all(q_forward[:30] < 0.5))
        assert (np.all(q_forward[40:] > 0.5))
        q_backward = msm.committor_backward(a, b)
        assert (q_backward[a] == 1)
        assert (q_backward[b] == 0)
        assert (np.all(q_backward[:30] > 0.5))
        assert (np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(msm.nstates)))

    def test_committor(self):
        self._committor(self.msmrev)
        self._committor(self.msm)

    def _mfpt(self, msm):
        a = 16
        b = 48
        t = msm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        if msm.is_reversible:
            assert (np.abs(t - 872.69132618104049) < 1e-6)
        else:
            assert (np.abs(t - 872.07000910307738) < 1e-6)

    def test_mfpt(self):
        self._mfpt(self.msmrev)
        self._mfpt(self.msm)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def _pcca_assignment(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            ass = msm.metastable_assignments
            # test: number of states
            assert (len(ass) == msm.nstates)
            # test: should be 0 or 1
            assert (np.all(ass >= 0))
            assert (np.all(ass <= 1))
            # should be equal (zero variance) within metastable sets
            assert (np.std(ass[:30]) == 0)
            assert (np.std(ass[40:]) == 0)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)
                ass = msm.metastable_assignments

    def test_pcca_assignment(self):
        self._pcca_assignment(self.msmrev)
        self._pcca_assignment(self.msm)

    def _pcca_distributions(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            pccadist = msm.metastable_distributions
            # should be right size
            assert (np.all(pccadist.shape == (2, msm.nstates)))
            # should be nonnegative
            assert (np.all(pccadist >= 0))
            # should roughly add up to stationary:
            ds = pccadist[0] + pccadist[1]
            ds /= ds.sum()
            assert (np.max(np.abs(ds - msm.stationary_distribution)) < 0.001)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)
                pccadist = msm.metastable_distributions

    def test_pcca_distributions(self):
        self._pcca_distributions(self.msmrev)
        self._pcca_distributions(self.msm)

    def _pcca_memberships(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            M = msm.metastable_memberships
            # should be right size
            assert (np.all(M.shape == (msm.nstates, 2)))
            # should be nonnegative
            assert (np.all(M >= 0))
            # should add up to one:
            assert (np.allclose(np.sum(M, axis=1), np.ones(msm.nstates)))
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)
                M = msm.metastable_memberships

    def test_pcca_memberships(self):
        self._pcca_memberships(self.msmrev)
        self._pcca_memberships(self.msm)

    def _pcca_sets(self, msm):
        if msm.is_reversible:
            msm.pcca(2)
            S = msm.metastable_sets
            assignment = msm.metastable_assignments
            # should coincide with assignment
            for i, s in enumerate(S):
                for j in range(len(s)):
                    assert (assignment[s[j]] == i)
        else:
            with self.assertRaises(ValueError):
                msm.pcca(2)
                S = msm.metastable_sets
                assignment = msm.metastable_assignments

    def test_pcca_sets(self):
        self._pcca_sets(self.msmrev)
        self._pcca_sets(self.msm)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, msm):
        e = msm.expectation(range(msm.nstates))
        # approximately equal for both
        assert (np.abs(e - 31.73) < 0.01)

    def test_expectation(self):
        self._expectation(self.msmrev)
        self._expectation(self.msm)

    def _correlation(self, msm):
        # raise assertion error because size is wrong:
        maxtime = 100000
        a = [1, 2, 3]
        with self.assertRaises(AssertionError):
            msm.correlation(a, 1)
        # should decrease
        a = range(msm.nstates)
        times, corr1 = msm.correlation(a, maxtime=maxtime)
        assert (len(corr1) == maxtime / msm._tau)
        assert (len(times) == maxtime / msm._tau)
        assert (corr1[0] > corr1[-1])
        a = range(msm.nstates)
        times, corr2 = msm.correlation(a, a, maxtime=maxtime, )
        # should be identical to autocorr
        assert (np.allclose(corr1, corr2))
        # Test: should be increasing in time
        b = range(msm.nstates)[::-1]
        times, corr3 = msm.correlation(a, b, maxtime=maxtime, )
        assert (len(times) == maxtime / msm._tau)
        assert (len(corr3) == maxtime / msm._tau)
        assert (corr3[0] < corr3[-1])

    def test_correlation(self):
        self._correlation(self.msmrev)
        self._correlation(self.msm)

    def _relaxation(self, msm):
        pi_perturbed = (msm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = range(msm.nstates)
        maxtime = 100000
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=maxtime)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = msm.relaxation(pi_perturbed, a, maxtime=maxtime)
        # should relax
        assert (len(times) == maxtime / msm._tau)
        assert (len(rel2) == maxtime / msm._tau)
        assert (rel2[0] < rel2[-1])

    def test_relaxation(self):
        self._relaxation(self.msmrev)
        self._relaxation(self.msm)

    def _fingerprint_correlation(self, msm):
        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_correlation(a, 1)
            # should decrease
            a = range(self.msm.nstates)
            fp1 = msm.fingerprint_correlation(a)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales()))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            # identical call
            b = range(msm.nstates)
            fp2 = msm.fingerprint_correlation(a, b)
            assert (np.allclose(fp1[0], fp2[0]))
            assert (np.allclose(fp1[1], fp2[1]))
            # should be - of the above, apart from the first
            b = range(msm.nstates)[::-1]
            fp3 = msm.fingerprint_correlation(a, b)
            assert (np.allclose(fp1[0], fp3[0]))
            assert (np.allclose(fp1[1][1:], -fp3[1][1:]))
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = range(self.msm.nstates)
                msm.fingerprint_correlation(a)
            with self.assertRaises(ValueError):
                a = range(self.msm.nstates)
                b = range(msm.nstates)
                msm.fingerprint_correlation(a, b)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.msmrev)
        self._fingerprint_correlation(self.msm)

    def _fingerprint_relaxation(self, msm):
        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a)
            # equilibrium relaxation should be constant
            a = range(msm.nstates)
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales()))
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            pi_perturbed = (msm.stationary_distribution ** 2)
            pi_perturbed /= pi_perturbed.sum()
            fp2 = msm.fingerprint_relaxation(pi_perturbed, a)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp2[0][1:], msm.timescales()))
            # dynamical amplitudes should be significant because we are not in equilibrium
            assert (np.max(np.abs(fp2[1][1:])) > 0.1)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = range(self.msm.nstates)
                msm.fingerprint_relaxation(msm.stationary_distribution, a)
            with self.assertRaises(ValueError):
                pi_perturbed = (msm.stationary_distribution ** 2)
                pi_perturbed /= pi_perturbed.sum()
                a = range(self.msm.nstates)
                msm.fingerprint_relaxation(pi_perturbed, a)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.msmrev)
        self._fingerprint_relaxation(self.msm)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def _active_state_indexes(self, msm):
        I = msm.active_state_indexes
        assert (len(I) == msm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt

        hist = dt.count_states(msm.discrete_trajectories_full)
        # number of frames should match on active subset
        A = msm.active_set
        for i in range(A.shape[0]):
            assert (I[i].shape[0] == hist[A[i]])
            assert (I[i].shape[1] == 2)

    def test_active_state_indexes(self):
        self._active_state_indexes(self.msmrev)
        self._active_state_indexes(self.msm)

    def _generate_traj(self, msm):
        T = 10
        gt = msm.generate_traj(T)
        # Test: should have the right dimension
        assert (np.all(gt.shape == (T, 2)))
        # itraj should be right
        assert (np.all(gt[:, 0] == 0))

    def test_generate_traj(self):
        self._generate_traj(self.msmrev)
        self._generate_traj(self.msm)

    def _sample_by_state(self, msm):
        nsample = 100
        ss = msm.sample_by_state(nsample)
        # must have the right size
        assert (len(ss) == msm.nstates)
        # must be correctly assigned
        dtraj_active = msm.discrete_trajectories_active[0]
        for i, samples in enumerate(ss):
            # right shape
            assert (np.all(samples.shape == (nsample, 2)))
            for row in samples:
                assert (row[0] == 0)  # right trajectory
                assert (dtraj_active[row[1]] == i)

    def test_sample_by_state(self):
        self._sample_by_state(self.msmrev)
        self._sample_by_state(self.msm)

    def _trajectory_weights(self, msm):
        W = msm.trajectory_weights()
        # should sum to 1
        assert (np.abs(np.sum(W[0]) - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.msmrev)
        self._trajectory_weights(self.msm)

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def _two_state_kinetics(self, msm):
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = msm.eigenvectors_left()[1, :]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = msm.mfpt(core1, core2)
        t21 = msm.mfpt(core2, core1)
        # relaxation time
        t2 = msm.timescales()[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        assert (np.abs(k2 - ksum) < 0.001)

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.msmrev)
        self._two_state_kinetics(self.msm)


if __name__ == "__main__":
    unittest.main()