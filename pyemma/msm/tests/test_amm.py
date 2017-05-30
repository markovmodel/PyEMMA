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


r"""Unit test for the AMM module

.. moduleauthor:: S. Olsson <solsson AT zedat DOT fu DASH berlin DOT de> 

"""

from __future__ import absolute_import
import unittest

import numpy as np
import scipy.sparse
import warnings

from msmtools.generation import generate_traj
from msmtools.estimation import count_matrix, largest_connected_set, largest_connected_submatrix, transition_matrix
from msmtools.analysis import stationary_distribution, timescales
from pyemma.util.numeric import assert_allclose
from pyemma.msm.tests.birth_death_chain import BirthDeathChain
from pyemma.msm import estimate_augmented_markov_model, AugmentedMarkovModel, estimate_markov_model 
from six.moves import range


class TestAMMSimple(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(0xDEADBEEF)

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

        self.k = 3
        """ Predictions and experimental data """
        self.E = np.vstack((np.linspace(-0.1, 1., 7), np.linspace(1.5, -0.1, 7))).T
        self.m = np.array([0.0, 0.0])
        self.w = np.array([2.0, 2.5])
        self.sigmas = 1./np.sqrt(2)/np.sqrt(self.w)

        """ Feature trajectory """
        self.ftraj = self.E[self.dtraj, :]

        self.AMM = AugmentedMarkovModel(E = self.E, m = self.m, w = self.w)
        self.AMM.estimate([self.dtraj])

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_AMM(self):
        """ self-consistency, explicit class instantiation/estimation and convienence function """
        amm = estimate_augmented_markov_model([self.dtraj], [self.ftraj], self.tau, self.m, self.sigmas)
        assert_allclose(self.dtraj, amm.discrete_trajectories_full[0])
        self.assertEqual(self.tau, amm.lagtime)
        self.assertTrue(np.allclose(self.E, amm.E))
        self.assertTrue(np.allclose(self.m, amm.m))
        self.assertTrue(np.allclose(self.w, amm.w))
        self.assertTrue(np.allclose(self.AMM.P, amm.P))
        self.assertTrue(np.allclose(self.AMM.pi, amm.pi))
        self.assertTrue(np.allclose(self.AMM.lagrange, amm.lagrange))

class TestAMMDoubleWell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyemma.datasets
        cls.dtraj = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10
        cls.E_ = np.linspace(0.01, 2.*np.pi, 66).reshape(-1,1)**(0.5)    
        cls.m = np.array([1.9]) 
        cls.w = np.array([2.0]) 
        cls.sigmas = 1./np.sqrt(2)/np.sqrt(cls.w)
        _sd = list(set(cls.dtraj))
        
        cls.ftraj = cls.E_[[_sd.index(d) for d in cls.dtraj], :]
        cls.tau = 10
        cls.amm = estimate_augmented_markov_model([cls.dtraj], [cls.ftraj], cls.tau, cls.m, cls.sigmas)


    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def _lagtime(self, amm):
        assert (amm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.amm)

    def _active_set(self, amm):
        # should always be <= full set
        assert (len(amm.active_set) <= self.amm.nstates_full)
        # should be length of nstates
        assert (len(amm.active_set) == self.amm.nstates)

    def test_active_set(self):
        self._active_set(self.amm)

    def _largest_connected_set(self, amm):
        lcs = amm.largest_connected_set
        # identical to first connected set
        assert (np.all(lcs == amm.connected_sets[0]))
        # LARGEST: identical to active set
        assert (np.all(lcs == amm.active_set))

    def test_largest_connected_set(self):
        self._largest_connected_set(self.amm)

    def _nstates(self, amm):
        # should always be <= full
        assert (amm.nstates <= amm.nstates_full)
        # THIS DATASET:
        assert (amm.nstates == 66)

    def test_nstates(self):
        self._nstates(self.amm)

    def _connected_sets(self, amm):
        cs = amm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == amm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.amm)

    def _connectivity(self, amm):
        # HERE:
        assert (amm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.amm)

    def _count_matrix_active(self, amm):
        C = amm.count_matrix_active
        assert (np.all(C.shape == (amm.nstates, amm.nstates)))

    def test_count_matrix_active(self):
        self._count_matrix_active(self.amm)

    def _count_matrix_full(self, amm):
        C = amm.count_matrix_full
        assert (np.all(C.shape == (amm.nstates_full, amm.nstates_full)))

    def test_count_matrix_full(self):
        self._count_matrix_full(self.amm)

    def _discrete_trajectories_full(self, amm):
        assert (np.all(self.dtraj == amm.discrete_trajectories_full[0]))

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.amm)

    def _discrete_trajectories_active(self, amm):
        dta = amm.discrete_trajectories_active
        # HERE
        assert (len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert (dta[0][0] < self.dtraj[0])

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.amm)

    def _timestep(self, amm):
        assert (amm.timestep_model.startswith('1'))
        assert (amm.timestep_model.endswith('step'))

    def test_timestep(self):
        self._timestep(self.amm)

    def _transition_matrix(self, amm):
        P = amm.transition_matrix
        # should be ndarray by default
        # assert (isinstance(P, np.ndarray))
        assert (isinstance(P, np.ndarray) or isinstance(P, scipy.sparse.csr_matrix))
        # shape
        assert (np.all(P.shape == (amm.nstates, amm.nstates)))
        # test transition matrix properties
        import msmtools.analysis as msmana

        assert (msmana.is_transition_matrix(P))
        assert (msmana.is_connected(P))
        # REVERSIBLE
        if amm.is_reversible:
            assert (msmana.is_reversible(P))

    def test_transition_matrix(self):
        self._transition_matrix(self.amm)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_count_fraction(self, amm):
        # should always be a fraction
        assert (0.0 <= amm.active_count_fraction <= 1.0)
        # special case for this data set:
        assert (amm.active_count_fraction == 1.0)

    def test_active_count_fraction(self):
        self._active_count_fraction(self.amm)

    def _active_state_fraction(self, amm):
        # should always be a fraction
        assert (0.0 <= amm.active_state_fraction <= 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.amm)

    def _effective_count_matrix(self, amm):
        Ceff = amm.effective_count_matrix
        assert (np.all(Ceff.shape == (amm.nstates, amm.nstates)))

    def test_effective_count_matrix(self):
        self._effective_count_matrix(self.amm)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, amm):
        mu = amm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)

    def test_statdist(self):
        self._statdist(self.amm)

    def _eigenvalues(self, amm):
        if not amm.is_sparse:
            ev = amm.eigenvalues()
        else:
            k = 4
            ev = amm.eigenvalues(k)
        # stochasticity
        assert (np.max(np.abs(ev)) <= 1 + 1e-12)
        # irreducible
        assert (np.max(np.abs(ev[1:])) < 1)
        # ordered?
        evabs = np.abs(ev)
        for i in range(0, len(evabs) - 1):
            assert (evabs[i] >= evabs[i + 1])
        # REVERSIBLE:
        if amm.is_reversible:
            assert (np.all(np.isreal(ev)))

    def test_eigenvalues(self):
        self._eigenvalues(self.amm)

    def _eigenvectors_left(self, amm):
        if not amm.is_sparse:
            L = amm.eigenvectors_left() 
            k = amm.nstates
        else:
            k = 4
            L = amm.eigenvectors_left(k)
        # shape should be right
        assert (np.all(L.shape == (k, amm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0, :]
        err = amm.stationary_distribution - l1
        assert (np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert (np.allclose(np.sum(L[1:, :], axis=1), np.zeros(k - 1)))
        # REVERSIBLE:
        if amm.is_reversible:
            assert (np.all(np.isreal(L)))

    def test_eigenvectors_left(self):
        self._eigenvectors_left(self.amm)

    def _eigenvectors_right(self, amm):
        if not amm.is_sparse:
            R = amm.eigenvectors_right()
            k = amm.nstates
        else:
            k = 4
            R = amm.eigenvectors_right(k)
        # shape should be right
        assert (np.all(R.shape == (amm.nstates, k)))
        # should be all ones
        r1 = R[:, 0]
        assert (np.allclose(r1, np.ones(amm.nstates)))
        # REVERSIBLE:
        if amm.is_reversible:
            assert (np.all(np.isreal(R)))

    def test_eigenvectors_right(self):
        self._eigenvectors_right(self.amm)

    def _eigenvectors_RDL(self, amm):
        if not amm.is_sparse:
            R = amm.eigenvectors_right()
            D = np.diag(amm.eigenvalues())
            L = amm.eigenvectors_left()
            # orthogonality constraint
            assert (np.allclose(np.dot(R, L), np.eye(amm.nstates)))
            # REVERSIBLE: also true for LR because reversible matrix
            if amm.is_reversible:
                assert (np.allclose(np.dot(L, R), np.eye(amm.nstates)))
            # recover transition matrix
            assert (np.allclose(np.dot(R, np.dot(D, L)), amm.transition_matrix))

        else:
            k = 4
            R = amm.eigenvectors_right(k)
            D = np.diag(amm.eigenvalues(k))
            L = amm.eigenvectors_left(k)
            """Orthoginality"""
            assert (np.allclose(np.dot(L, R), np.eye(k)))
            """Reversibility"""
            if amm.is_reversible:
                mu = amm.stationary_distribution
                L_mu = mu[:,np.newaxis] * R
                assert (np.allclose(np.dot(L_mu.T, R), np.eye(k)))


    def test_eigenvectors_RDL(self):
        self._eigenvectors_RDL(self.amm)

    def _timescales(self, amm):
        if not amm.is_sparse:
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales()
            else:
                ts = amm.timescales()
        else:
            k = 4
            if not amm.is_reversible:
                with warnings.catch_warnings(record=True) as w:
                    ts = amm.timescales(k)
            else:
                ts = amm.timescales(k)

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if amm.is_reversible:
            ts_ref = np.array([ 299.11,    8.58,    5.1 ])
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)
        else:
            ts_ref = np.array([ 299.11,    8.58,    5.1 ])
            # HERE:
            np.testing.assert_almost_equal(ts[:3], ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.amm)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def _committor(self, amm):
        a = 16
        b = 48
        q_forward = amm.committor_forward(a, b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)
        assert (np.all(q_forward[:30] < 0.5))
        assert (np.all(q_forward[40:] > 0.5))
        q_backward = amm.committor_backward(a, b)
        assert (q_backward[a] == 1)
        assert (q_backward[b] == 0)
        assert (np.all(q_backward[:30] > 0.5))
        assert (np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        if amm.is_reversible:
            assert (np.allclose(q_forward + q_backward, np.ones(amm.nstates)))

    def test_committor(self):
        self._committor(self.amm)

    def _mfpt(self, amm):
        a = 16
        b = 48
        t = amm.mfpt(a, b)
        assert (t > 0)
        # HERE:
        np.testing.assert_allclose(t, 709.76, rtol=1e-3, atol=1e-6)

    def test_mfpt(self):
        self._mfpt(self.amm)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def _pcca_assignment(self, amm):
        if amm.is_reversible:
            amm.pcca(2)
            ass = amm.metastable_assignments
            # test: number of states
            assert (len(ass) == amm.nstates)
            # test: should be 0 or 1
            assert (np.all(ass >= 0))
            assert (np.all(ass <= 1))
            # should be equal (zero variance) within metastable sets
            assert (np.std(ass[:30]) == 0)
            assert (np.std(ass[40:]) == 0)
        else:
            with self.assertRaises(ValueError):
                amm.pcca(2)

    def test_pcca_assignment(self):
        self._pcca_assignment(self.amm)
        

    def _pcca_distributions(self, amm):
        if amm.is_reversible:
            amm.pcca(2)
            pccadist = amm.metastable_distributions
            # should be right size
            assert (np.all(pccadist.shape == (2, amm.nstates)))
            # should be nonnegative
            assert (np.all(pccadist >= 0))
            # should roughly add up to stationary:
            # this will not hold for AMMs?
            #ds = pccadist[0] + pccadist[1]
            #ds /= ds.sum()
            #assert (np.max(np.abs(ds - amm.stationary_distribution)) < 0.001)
        else:
            with self.assertRaises(ValueError):
                amm.pcca(2)

    def test_pcca_distributions(self):
        self._pcca_distributions(self.amm)
        

    def _pcca_memberships(self, amm):
        if amm.is_reversible:
            amm.pcca(2)
            M = amm.metastable_memberships
            # should be right size
            assert (np.all(M.shape == (amm.nstates, 2)))
            # should be nonnegative
            assert (np.all(M >= 0))
            # should add up to one:
            assert (np.allclose(np.sum(M, axis=1), np.ones(amm.nstates)))
        else:
            with self.assertRaises(ValueError):
                amm.pcca(2)

    def test_pcca_memberships(self):
        self._pcca_memberships(self.amm)

    def _pcca_sets(self, amm):
        if amm.is_reversible:
            amm.pcca(2)
            S = amm.metastable_sets
            assignment = amm.metastable_assignments
            # should coincide with assignment
            for i, s in enumerate(S):
                for j in range(len(s)):
                    assert (assignment[s[j]] == i)
        else:
            with self.assertRaises(ValueError):
                amm.pcca(2)

    def test_pcca_sets(self):
        self._pcca_sets(self.amm)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def _expectation(self, amm):
        e = amm.expectation(list(range(amm.nstates)))
        # approximately equal for both
        assert (np.abs(e - 34.92) < 0.01)

    def test_expectation(self):
        self._expectation(self.amm)

    def _correlation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates            
        # raise assertion error because size is wrong:
        maxtime = 100000
        a = [1, 2, 3]
        with self.assertRaises(AssertionError):
            amm.correlation(a, 1)
        # should decrease
        a = list(range(amm.nstates))
        times, corr1 = amm.correlation(a, maxtime=maxtime)
        assert (len(corr1) == maxtime / amm.lagtime)
        assert (len(times) == maxtime / amm.lagtime)
        assert (corr1[0] > corr1[-1])
        a = list(range(amm.nstates))
        times, corr2 = amm.correlation(a, a, maxtime=maxtime, k=k)
        # should be identical to autocorr
        assert (np.allclose(corr1, corr2))
        # Test: should be increasing in time
        b = list(range(amm.nstates))[::-1]
        times, corr3 = amm.correlation(a, b, maxtime=maxtime, )
        assert (len(times) == maxtime / amm.lagtime)
        assert (len(corr3) == maxtime / amm.lagtime)
        assert (corr3[0] < corr3[-1])

    def test_correlation(self):
        self._correlation(self.amm)

    def _relaxation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates            
        pi_perturbed = (amm.stationary_distribution ** 2)
        pi_perturbed /= pi_perturbed.sum()
        a = list(range(amm.nstates))[::-1]
        maxtime = 100000
        times, rel1 = amm.relaxation(amm.stationary_distribution, a, maxtime=maxtime, k=k)
        # should be constant because we are in equilibrium
        assert (np.allclose(rel1 - rel1[0], np.zeros((np.shape(rel1)[0]))))
        times, rel2 = amm.relaxation(pi_perturbed, a, maxtime=maxtime, k=k)
        # should relax
        assert (len(times) == maxtime / amm.lagtime)
        assert (len(rel2) == maxtime / amm.lagtime)
        assert (rel2[0] < rel2[-1])

    def test_relaxation(self):
        self._relaxation(self.amm)

    def _fingerprint_correlation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates       

        if amm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                amm.fingerprint_correlation(a, 1, k=k)
            # should decrease
            a = list(range(self.amm.nstates))
            fp1 = amm.fingerprint_correlation(a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], amm.timescales(k-1)))
            # all amplitudes nonnegative (for autocorrelation)
            assert (np.all(fp1[1][:] >= 0))
            # identical call
            b = list(range(amm.nstates))
            fp2 = amm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp2[0]))
            assert (np.allclose(fp1[1], fp2[1]))
            # should be - of the above, apart from the first
            b = list(range(amm.nstates))[::-1]
            fp3 = amm.fingerprint_correlation(a, b, k=k)
            assert (np.allclose(fp1[0], fp3[0]))
            assert (np.allclose(fp1[1][1:], -fp3[1][1:]))
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = list(range(self.amm.nstates))
                amm.fingerprint_correlation(a, k=k)
            with self.assertRaises(ValueError):
                a = list(range(self.amm.nstates))
                b = list(range(amm.nstates))
                amm.fingerprint_correlation(a, b, k=k)

    def test_fingerprint_correlation(self):
        self._fingerprint_correlation(self.amm)

    def _fingerprint_relaxation(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates       

        if amm.is_reversible:
            # raise assertion error because size is wrong:
            a = [1, 2, 3]
            with self.assertRaises(AssertionError):
                amm.fingerprint_relaxation(amm.stationary_distribution, a, k=k)
            # equilibrium relaxation should be constant
            a = list(range(amm.nstates))
            fp1 = amm.fingerprint_relaxation(amm.stationary_distribution, a, k=k)
            # first timescale is infinite
            assert (fp1[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp1[0][1:], amm.timescales(k-1)))
            # dynamical amplitudes should be near 0 because we are in equilibrium
            assert (np.max(np.abs(fp1[1][1:])) < 1e-10)
            # off-equilibrium relaxation
            pi_perturbed = (amm.stationary_distribution ** 2)
            pi_perturbed /= pi_perturbed.sum()
            fp2 = amm.fingerprint_relaxation(pi_perturbed, a, k=k)
            # first timescale is infinite
            assert (fp2[0][0] == np.inf)
            # next timescales are identical to timescales:
            assert (np.allclose(fp2[0][1:], amm.timescales(k-1)))
            # dynamical amplitudes should be significant because we are not in equilibrium
            assert (np.max(np.abs(fp2[1][1:])) > 0.1)
        else:  # raise ValueError, because fingerprints are not defined for nonreversible
            with self.assertRaises(ValueError):
                a = list(range(self.amm.nstates))
                amm.fingerprint_relaxation(amm.stationary_distribution, a, k=k)
            with self.assertRaises(ValueError):
                pi_perturbed = (amm.stationary_distribution ** 2)
                pi_perturbed /= pi_perturbed.sum()
                a = list(range(self.amm.nstates))
                amm.fingerprint_relaxation(pi_perturbed, a)

    def test_fingerprint_relaxation(self):
        self._fingerprint_relaxation(self.amm)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def _active_state_indexes(self, amm):
        I = amm.active_state_indexes
        assert (len(I) == amm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt

        hist = dt.count_states(amm.discrete_trajectories_full)
        # number of frames should match on active subset
        A = amm.active_set
        for i in range(A.shape[0]):
            assert (I[i].shape[0] == hist[A[i]])
            assert (I[i].shape[1] == 2)

    def test_active_state_indexes(self):
        self._active_state_indexes(self.amm)

    def _generate_traj(self, amm):
        T = 10
        gt = amm.generate_traj(T)
        # Test: should have the right dimension
        assert (np.all(gt.shape == (T, 2)))
        # itraj should be right
        assert (np.all(gt[:, 0] == 0))

    def test_generate_traj(self):
        self._generate_traj(self.amm)

    def _sample_by_state(self, amm):
        nsample = 100
        ss = amm.sample_by_state(nsample)
        # must have the right size
        assert (len(ss) == amm.nstates)
        # must be correctly assigned
        dtraj_active = amm.discrete_trajectories_active[0]
        for i, samples in enumerate(ss):
            # right shape
            assert (np.all(samples.shape == (nsample, 2)))
            for row in samples:
                assert (row[0] == 0)  # right trajectory
                assert (dtraj_active[row[1]] == i)

    def test_sample_by_state(self):
        self._sample_by_state(self.amm)

    def _trajectory_weights(self, amm):
        W = amm.trajectory_weights()
        # should sum to 1
        assert (np.abs(np.sum(W[0]) - 1.0) < 1e-6)

    def test_trajectory_weights(self):
        self._trajectory_weights(self.amm)

    def test_simulate_MSM(self):
        amm = self.amm
        N=400
        start=1
        traj = amm.simulate(N=N, start=start)
        assert (len(traj) <= N)
        assert (len(np.unique(traj)) <= len(amm.transition_matrix))
        assert (start == traj[0])

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def _two_state_kinetics(self, amm):
        if amm.is_sparse:
            k = 4
        else:
            k = amm.nstates
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = amm.eigenvectors_left(k)[1, :]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = amm.mfpt(core1, core2)
        t21 = amm.mfpt(core2, core1)
        # relaxation time
        t2 = amm.timescales(k)[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and
        # therefore underestimate rates
        ksum = 1.0 / t12 + 1.0 / t21
        k2 = 1.0 / t2
        assert (np.abs(k2 - ksum) < 0.001)

    def test_two_state_kinetics(self):
        self._two_state_kinetics(self.amm)


if __name__ == "__main__":
    unittest.main()
