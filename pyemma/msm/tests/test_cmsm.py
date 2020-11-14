
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


r"""Unit test for the core set MSM module

"""

import unittest
import warnings

import numpy as np
import scipy.sparse

import pyemma
from pyemma.msm import estimate_markov_model


class TestCMSMRevPi(unittest.TestCase):
    r"""Checks if the MLMSM correctly handles the active set computation
    if a stationary distribution is given"""

    def test_valid_stationary_vector(self):
        dtraj = np.array([0, 0, 1, 0, 1, 2])
        pi_invalid = np.array([0.1, 0.9, 0.0])
        pi_valid = np.array([0.1, 0.9])
        core_set = np.array([0, 1])
        msm = estimate_markov_model(dtraj, 1, statdist=pi_valid, core_set=core_set)
        self.assertTrue(np.all(msm.active_set==core_set))
        np.testing.assert_array_equal(msm.pi, pi_valid)
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj, 1, statdist=pi_invalid, core_set=core_set)

    def test_valid_trajectory(self):
        pi = np.array([0.1, 0.9])
        dtraj_invalid = np.array([1, 1, 1, 1, 1, 1, 1])
        dtraj_valid = np.array([0, 2, 0, 2, 2, 0, 1, 1])
        core_set = [0, 2]
        msm = estimate_markov_model(dtraj_valid, 1, statdist=pi, core_set=core_set)
        self.assertTrue(np.all(msm.active_set==np.array(core_set)))
        np.testing.assert_array_equal(msm.pi, pi)
        with self.assertRaises(ValueError):
            estimate_markov_model(dtraj_invalid, 1, statdist=pi, core_set=core_set)


class TestCMSMDoubleWell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.datasets
        cls.core_set = [34, 65]

        cls.dtraj = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10
        nu = 1.*np.bincount(cls.dtraj)[cls.core_set]
        cls.statdist = nu/nu.sum()

        cls.tau = 10
        maxerr = 1e-12

        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.msmrev = estimate_markov_model(cls.dtraj, cls.tau ,maxerr=maxerr, core_set=cls.core_set)
            cls.msmrevpi = estimate_markov_model(cls.dtraj, cls.tau,maxerr=maxerr,
                                                 statdist=cls.statdist, core_set=cls.core_set)
            cls.msm = estimate_markov_model(cls.dtraj, cls.tau, reversible=False, maxerr=maxerr,
                                            core_set=cls.core_set)


    # ---------------------------------
    # SCORE
    # ---------------------------------
    def _score(self, msm):
        # check estimator args are not overwritten, if default arguments are used.
        msm.score_k = 2  # default of 10 is too high for 2 state system
        old_score_k = msm.score_k
        old_score_method = msm.score_method
        dtrajs_test = self.dtraj[80000:]
        msm.score(dtrajs_test)
        assert msm.score_k == old_score_k
        assert msm.score_method == old_score_method
        s1 = msm.score(dtrajs_test, score_method='VAMP1', score_k=2)
        assert msm.score_k == 2
        assert msm.score_method == 'VAMP1'
        assert 1.0 <= s1 <= 2.0 + 1e-15

        s2 = msm.score(dtrajs_test, score_method='VAMP2', score_k=2)
        assert 1.0 <= s2 <= 2.0 + 1e-15
        assert msm.score_k == 2
        assert msm.score_method == 'VAMP2'
        # se = msm.score(dtrajs_test, score_method='VAMPE', score_k=2)
        # se_inf = msm.score(dtrajs_test, score_method='VAMPE', score_k=None)

    def test_score(self):
        self._score(self.msmrev)
        self._score(self.msmrevpi)
        self._score(self.msm)


    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_reversible(self):
        # NONREVERSIBLE
        assert self.msmrev.is_reversible
        assert self.msmrevpi.is_reversible

        # REVERSIBLE
        assert not self.msm.is_reversible

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msmrevpi)
        self._lagtime(self.msm)

    def _active_set(self, msm):
        # should always be <= full set
        self.assertLessEqual(len(msm.active_set), self.msm.nstates_full)
        # should be length of nstates
        self.assertEqual(len(msm.active_set), self.msm.nstates)

    def test_active_set(self):
        self._active_set(self.msmrev)
        self._active_set(self.msmrevpi)
        self._active_set(self.msm)

    def _largest_connected_set(self, msm):
        lcs = msm.largest_connected_set
        # identical to first connected set
        assert (np.all(lcs == msm.connected_sets[0]))
        # LARGEST: identical to active set
        assert (np.all(lcs == msm.active_set))

    def test_largest_connected_set(self):
        self._largest_connected_set(self.msmrev)
        self._largest_connected_set(self.msmrevpi)
        self._largest_connected_set(self.msm)

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 2)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msmrevpi)
        self._nstates(self.msm)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msmrevpi)
        self._connected_sets(self.msm)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msmrevpi)
        self._connectivity(self.msm)

    def _count_matrix_active(self, msm):
        C = msm.count_matrix_active
        assert (np.all(C.shape == (msm.nstates, msm.nstates)))

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msmrevpi)
        self._count_matrix_active(self.msm)

    def _count_matrix_full(self, msm):
        C = msm.count_matrix_full
        assert (np.all(C.shape == (msm.nstates_full, msm.nstates_full)))

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msmrevpi)
        self._count_matrix_full(self.msm)

    def _discrete_trajectories_full(self, msm):
        # this only checks for states originally in core set as dtraj is
        # rewritten depending on coring method

        _dtraj_cored = np.array([d if d in self.core_set else -1 for d in self.dtraj])
        _assigned = np.where(_dtraj_cored >= 0)
        assert (np.all(_dtraj_cored[_assigned] == msm.discrete_trajectories_full[0][_assigned]))

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msmrevpi)
        self._discrete_trajectories_full(self.msm)

    def _discrete_trajectories_active(self, msm):
        dta = msm.discrete_trajectories_active
        # HERE
        assert (len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert (dta[0][0] < self.dtraj[0])

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msmrevpi)
        self._discrete_trajectories_active(self.msm)

    def _timestep(self, msm):
        assert (msm.timestep_model.startswith('1'))
        assert (msm.timestep_model.endswith('step'))

    def test_timestep(self):
        self._timestep(self.msmrev)
        self._timestep(self.msmrevpi)
        self._timestep(self.msm)

    def _dt_model(self, msm):
        from pyemma.util.units import TimeUnit
        tu = TimeUnit("1 step").get_scaled(self.msm.lag)
        self.assertEqual(msm.dt_model, tu)

    def test_dt_model(self):
        self._dt_model(self.msmrev)
        self._dt_model(self.msmrevpi)
        self._dt_model(self.msm)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
        # assert (isinstance(P, np.ndarray))
        assert (isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        assert (np.all(P.shape == (msm.nstates, msm.nstates)))
        # test transition matrix properties
        import msmtools.analysis as msmana

        assert (msmana.is_transition_matrix(P))
        assert (msmana.is_connected(P))
        # REVERSIBLE
        if msm.is_reversible:
            assert (msmana.is_reversible(P))

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------
    def _active_count_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (msm.active_count_fraction == 1.0)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.msmrev)
        self._active_count_fraction(self.msmrevpi)
        self._active_count_fraction(self.msm)

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msmrevpi)
        self._active_state_fraction(self.msm)


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
        self._statdist(self.msmrevpi)
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
        self._eigenvalues(self.msmrevpi)
        self._eigenvalues(self.msm)

    def _eigenvectors_left(self, msm):

        L = msm.eigenvectors_left()
        k = msm.nstates

        # shape should be right
        assert (np.all(L.shape == (k, msm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0, :]
        err = msm.stationary_distribution - l1
        assert (np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert (np.allclose(np.sum(L[1:, :], axis=1), np.zeros(k - 1)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(L)))

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.msmrev)
        self._eigenvectors_left(self.msmrevpi)
        self._eigenvectors_left(self.msm)

    def _eigenvectors_right(self, msm):

        R = msm.eigenvectors_right()
        k = msm.nstates

        # shape should be right
        assert (np.all(R.shape == (msm.nstates, k)))
        # should be all ones
        r1 = R[:, 0]
        assert (np.allclose(r1, np.ones(msm.nstates)))
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.all(np.isreal(R)))

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.msmrev)
        self._eigenvectors_right(self.msmrevpi)
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
        self._eigenvectors_RDL(self.msmrevpi)
        self._eigenvectors_RDL(self.msm)

    def _timescales(self, msm):

        if not msm.is_reversible:
            with warnings.catch_warnings(record=True) as w:
                ts = msm.timescales()
        else:
            ts = msm.timescales()

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        # is there a better reference?
        ts_ref = np.array([360.])
        if msm.is_reversible:
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts, ts_ref, decimal=0)
        else:
            # HERE:
            np.testing.assert_almost_equal(ts, ts_ref, decimal=0)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, msm):
        a = 0
        b = 1
        q_forward = msm.committor_forward(a, b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)

        q_backward = msm.committor_backward(a, b)
        assert (q_backward[a] == 1)
        assert (q_backward[b] == 0)
        # REVERSIBLE:
        if msm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(msm.nstates)))

    def test_committor(self):
        self._committor(self.msmrev)
        self._committor(self.msm)


    def _mfpt(self, msm):
        a = 0
        b = 1
        t = msm.mfpt(a, b)
        assert (t > 0)
        # reference value?
        np.testing.assert_allclose(t, 739, rtol=1e-1, atol=1e-1)

    def test_mfpt(self):
        self._mfpt(self.msmrev)
        self._mfpt(self.msm)


    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, msm):
        e = msm.expectation([0, 85])
        self.assertLess(np.abs(e - 42.), 0.1)

    def test_expectation(self):
        self._expectation(self.msmrev)
        self._expectation(self.msm)

    def _correlation(self, msm):
        k = msm.nstates
        # raise assertion error because size is wrong:
        maxtime = 100000
        a = [1, 2, 3]
        with self.assertRaises(AssertionError):
            msm.correlation(a, 1)
        # should decrease
        a = list(range(msm.nstates))
        times, corr1 = msm.correlation(a, maxtime=maxtime)
        assert (len(corr1) == maxtime / msm.lagtime)
        assert (len(times) == maxtime / msm.lagtime)
        assert (corr1[0] > corr1[-1])
        a = list(range(msm.nstates))
        times, corr2 = msm.correlation(a, a, maxtime=maxtime, k=k)
        # should be identical to autocorr
        assert (np.allclose(corr1, corr2))
        # Test: should be increasing in time
        b = list(range(msm.nstates))[::-1]
        times, corr3 = msm.correlation(a, b, maxtime=maxtime, )
        assert (len(times) == maxtime / msm.lagtime)
        assert (len(corr3) == maxtime / msm.lagtime)
        assert (corr3[0] < corr3[-1])

    def test_correlation(self):
        self._correlation(self.msmrev)
        # self._correlation(self.msm)

    def _relaxation(self, msm):
        k = msm.nstates
        pi_perturbed = (msm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(msm.nstates))
        maxtime = 100000
        times, rel1 = msm.relaxation(msm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = msm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert (len(times) == maxtime / msm.lagtime)
        assert (len(rel2) == maxtime / msm.lagtime)
        self.assertLess(rel2[0], rel2[-1], msm)

    def test_relaxation(self):
        self._relaxation(self.msmrev)
        self._relaxation(self.msm)

    def _fingerprint_correlation(self, msm):

        k = msm.nstates

        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_correlation(a, 1, k=k)
            # should decrease
            a = list(range(self.msm.nstates))
            fp1 = msm.fingerprint_correlation(a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(k-1)))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            # identical call
            b = list(range(msm.nstates))
            fp2 = msm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp2[0]))
            assert (np.allclose(fp1[1], fp2[1]))
            # should be - of the above, apart from the first
            b = list(range(msm.nstates))[::-1]
            fp3 = msm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp3[0]))
            assert (np.allclose(fp1[1][1:], -fp3[1][1:]))
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                msm.fingerprint_correlation(a, k=k)
            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                b = list(range(msm.nstates))
                msm.fingerprint_correlation(a, b, k=k)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.msmrev)
        # TODO: 2-state MSM is not actually non-reversible
        #self._fingerprint_correlation(self.msm)

    def _fingerprint_relaxation(self, msm):
        k = msm.nstates

        if msm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # equilibrium relaxation should be constant
            a = list(range(msm.nstates))
            fp1 = msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], msm.timescales(k-1)))
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            pi_perturbed = msm.stationary_distribution + np.array([-.25, .25])
            pi_perturbed /= pi_perturbed.sum()
            fp2 = msm.fingerprint_relaxation(pi_perturbed, a, k=k)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp2[0][1:], msm.timescales(k-1)))
            # dynamical amplitudes should be significant because we are not in equilibrium
            assert (np.max(np.abs(fp2[1][1:])) > 0.1)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible

            with self.assertRaises(ValueError):
                a = list(range(self.msm.nstates))
                msm.fingerprint_relaxation(msm.stationary_distribution, a, k=k)
            with self.assertRaises(ValueError):
                pi_perturbed = (msm.stationary_distribution ** 2)
                pi_perturbed /= pi_perturbed.sum()
                a = list(range(self.msm.nstates))
                msm.fingerprint_relaxation(pi_perturbed, a)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.msmrev)
        # TODO: 2-state MSM is not actually non-reversible
        #self._fingerprint_relaxation(self.msm)

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
        self._active_state_indexes(self.msmrevpi)
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
        self._generate_traj(self.msmrevpi)
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
                self.assertEqual(dtraj_active[row[1]], i)

    def test_sample_by_state(self):
        self._sample_by_state(self.msmrev)
        self._sample_by_state(self.msmrevpi)
        self._sample_by_state(self.msm)

    def _trajectory_weights(self, msm):
        W = msm.trajectory_weights()
        # should sum to 1
        assert (np.abs(np.sum(W[0]) - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.msmrev)
        self._trajectory_weights(self.msmrevpi)
        self._trajectory_weights(self.msm)

    def test_simulate_MSM(self):
        msm = self.msm
        N=400
        start=1
        traj = msm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(msm.transition_matrix))
        assert (start == traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------
    def _two_state_kinetics(self, msm, eps=0.001):

        k = msm.nstates
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = msm.eigenvectors_left(k)[1, :]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = msm.mfpt(core1, core2)
        t21 = msm.mfpt(core2, core1)
        # relaxation time
        t2 = msm.timescales(k)[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        self.assertLess(np.abs(k2 - ksum), eps)

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.msmrev)
        self._two_state_kinetics(self.msmrevpi)
        self._two_state_kinetics(self.msm)


class TestCoreMSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyemma import datasets
        cls.dtraj = datasets.load_2well_discrete().dtraj_T100K_dt10

    def test_core(self):
        core_set = [15, 16, 17, 45, 46, 47]
        msm = pyemma.msm.estimate_markov_model(self.dtraj, lag=1, core_set=core_set)
        np.testing.assert_equal(msm.core_set, core_set)

        self.assertEqual(msm.n_cores, len(core_set))
        # check we only have core set states in the stored discrete trajectories.
        for d in msm.dtrajs_full:
            uniq = np.unique(d)
            assert len(np.setdiff1d(uniq, core_set)) == 0

    def test_indices_remapping_sample_by_dist(self):
        dtrajs = [[5, 5, 1, 0, 0, 1], [5, 1, 0, 1, 3], [0, 1, 2, 3]]
        desired_offsets = [2, 1, 0]
        msm = pyemma.msm.estimate_markov_model(dtrajs, lag=1, core_set=[0, 1, 2, 3])
        np.testing.assert_equal(msm.dtrajs_milestone_counting_offsets, desired_offsets)

        msm.pcca(2)
        samples = msm.sample_by_distributions(msm.metastable_distributions, 3)

        # check that non-core-set states are not drawn
        forbidden_traj_index_tuples = [(0, 0), (0, 1), (1, 0)]
        samples = [tuple(x) for x in np.vstack(samples)]
        for tpl in forbidden_traj_index_tuples:
            self.assertNotIn(tpl, samples)

    def test_indices_remapping_sample_by_state(self):
        dtrajs = [[5, 5, 1, 0, 0, 1], [5, 1, 0, 1, 3], [0, 1, 2, 3]]
        msm = pyemma.msm.estimate_markov_model(dtrajs, lag=1, core_set=[0, 1, 2, 3])

        samples = msm.sample_by_state(3)

        # check that non-core-set states are not drawn
        forbidden_traj_index_tuples = [(0, 0), (0, 1), (1, 0)]
        samples = [tuple(x) for x in np.vstack(samples)]
        for tpl in forbidden_traj_index_tuples:
            self.assertNotIn(tpl, samples)

    def test_compare2hmm(self):
        """test if estimated core set MSM is comparable to 2-state HMM; double-well"""

        cmsm = pyemma.msm.estimate_markov_model(self.dtraj, lag=5, core_set=[34, 65])
        hmm = pyemma.msm.estimate_hidden_markov_model(self.dtraj, nstates=2, lag=5)

        np.testing.assert_allclose(hmm.transition_matrix, cmsm.transition_matrix, rtol=.1, atol=1e-3)
        np.testing.assert_allclose(hmm.timescales()[0], cmsm.timescales()[0], rtol=.1)
        np.testing.assert_allclose(hmm.mfpt([0], [1]), cmsm.mfpt([0], [1]), rtol=.1)

    def test_compare2hmm_bayes(self):
        """test core set MSM with Bayesian sampling, compare ITS to 2-state BHMM; double-well"""

        cmsm = pyemma.msm.bayesian_markov_model(self.dtraj, lag=5, core_set=[34, 65], nsamples=20, count_mode='sliding')
        hmm = pyemma.msm.bayesian_hidden_markov_model(self.dtraj, 2, lag=5, nsamples=20)

        has_overlap = not (np.all(cmsm.sample_conf('timescales') < hmm.sample_conf('timescales')[0]) or
                           np.all(cmsm.sample_conf('timescales') > hmm.sample_conf('timescales')[1]))

        self.assertTrue(has_overlap, msg='Bayesian distributions of HMM and CMSM implied timescales have no overlap.')

    def test_last_core_counting(self):
        """test core set MSM with last visited core counting against a naive implementation"""

        n_states = 30
        n_traj = 10
        n_cores = 15
        dtrajs = [np.random.randint(0, n_states, size=1000) for _ in range(n_traj)]

        # have to ensure that highest state number is in core set
        # if state n_states-1 not in core_set, full count matrix becomes smaller
        # than naive implementation
        core_set = np.random.choice(np.arange(0, n_states-1), size=n_cores-1, replace=False)
        core_set = np.concatenate([core_set, [n_states-1]])
        assert np.unique(core_set).size == n_cores

        cmsm = pyemma.msm.estimate_markov_model(dtrajs, lag=1, core_set=core_set,
                                                count_mode='sample', reversible=False)

        def naive(dtrajs, core_set):
            import copy
            dtrajs = copy.deepcopy(dtrajs)
            nstates = np.concatenate(dtrajs).max() + 1
            cmat = np.zeros((nstates, nstates))
            newdiscretetraj = []
            for t, st in enumerate(dtrajs):
                oldmicro = None
                newtraj = []
                for f, micro in enumerate(st):
                    newmicro = None
                    for co in core_set:
                        if micro == co:
                            newmicro = micro
                            oldmicro = micro
                            break
                    if newmicro is None and oldmicro is not None:
                        newtraj.append(oldmicro)
                    elif newmicro is not None:
                        newtraj.append(newmicro)
                newdiscretetraj.append(np.array(newtraj, dtype=int))

            for d in newdiscretetraj:
                for oldmicro, newmicro in zip(d[:-1], d[1:]):
                    cmat[oldmicro, newmicro] += 1

            return newdiscretetraj, cmat

        expected_dtraj, expected_cmat = naive(dtrajs, core_set)
        np.testing.assert_equal(cmsm.dtrajs_full, expected_dtraj)
        np.testing.assert_equal(cmsm.count_matrix_full, expected_cmat)

    def test_not_implemented_raises(self):
        with self.assertRaises(RuntimeError) as e:
            pyemma.msm.bayesian_markov_model([0, 1, 0, 1, 0, 2, 2, 0], lag=1,
                                             core_set=[0, 1], count_mode='effective')
            self.assertIn('core set MSM with effective counting', e.exception.args[0])

        with self.assertRaises(NotImplementedError) as e:
            pyemma.msm.timescales_msm([0, 1, 0, 1, 0, 2, 2, 0], lags=[1, 2],
                                             core_set=[0, 1], errors='bayes')
            self.assertIn('does not support Bayesian error estimates for core set MSMs',
                          e.exception.args[0])

        with self.assertRaises(NotImplementedError) as e:
            pyemma.msm.estimate_markov_model([0, 1, 0, 1, 0, 2, 2, 0], lag=1,
                                             core_set=[0, 1], weights='oom')
            self.assertIn('Milestoning not implemented for OOMs', e.exception.args[0])

if __name__ == "__main__":
    unittest.main()
