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
from pyemma.util.numeric import assert_allclose, allclose_sparse
from pyemma.msm.ui.birth_death_chain import BirthDeathChain
from pyemma.msm import msm as markov_state_model

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
        q[2] = 1.0-10**(-b)
        q[4] = 10**(-b)
        p[2] = 10**(-b)
        p[4] = 1.0-10**(-b)

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
        assert_allclose(self.ts[1:], msm.timescales(self.k-1))

class TestMSMDoubleWellReversible(unittest.TestCase):

    def setUp(self):
        testpath = abspath(join(abspath(__file__), pardir)) + '/../../util/tests/data/'
        import pyemma.util.discrete_trajectories as dt
        self.dtraj = dt.read_discrete_trajectory(testpath+'2well_traj_100K.dat')
        self.tau = 10
        self.msm = markov_state_model(self.dtraj, self.tau)

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_compute(self):
        # should give warning
        with warnings.catch_warnings(record=True) as w:
            self.msm.compute()

    def test_computed(self):
        assert(self.msm.computed)

    def test_reversible(self):
        # REVERSIBLE
        assert(self.msm.is_reversible)

    def test_sparse(self):
        # default
        assert(not self.msm.is_sparse)

    def test_lagtime(self):
        assert(self.msm.lagtime == self.tau)

    def test_active_set(self):
        # should always be <= full set
        assert(len(self.msm.active_set) <= self.msm._n_full)
        # REVERSIBLE: should be range(msm.n_states)
        assert(len(self.msm.active_set) == self.msm.nstates)

    def test_largest_connected_set(self):
        lcs = self.msm.largest_connected_set
        # identical to first connected set
        assert(np.all(lcs == self.msm.connected_sets[0]))
        # REVERSIBLE: identical to active set
        assert(np.all(lcs == self.msm.active_set))

    def test_nstates(self):
        # should always be <= full
        assert(self.msm.nstates <= self.msm._n_full)
        # THIS DATASET:
        assert(self.msm.nstates == 66)

    def test_connected_sets(self):
        cs = self.msm.connected_sets
        assert(len(cs) >= 1)
        # REVERSIBLE:
        assert(np.all(cs[0] == self.msm.active_set))

    def test_connectivity(self):
        # HERE:
        assert(self.msm.connectivity == 'largest')

    def test_count_matrix_active(self):
        C = self.msm.count_matrix_active
        assert(np.all(C.shape == (self.msm.nstates,self.msm.nstates)))

    def test_count_matrix_full(self):
        C = self.msm.count_matrix_full
        assert(np.all(C.shape == (self.msm._n_full,self.msm._n_full)))

    def test_discrete_trajectories_full(self):
        assert(np.all(self.dtraj == self.msm.discrete_trajectories_full[0]))

    def test_discrete_trajectories_active(self):
        dta = self.msm.discrete_trajectories_active
        # HERE
        assert(len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert(dta[0][0] < self.dtraj[0])

    def test_timestep(self):
        assert(self.msm.timestep.startswith('1'))
        assert(self.msm.timestep.endswith('step'))

    def test_transition_matrix(self):
        P = self.msm.transition_matrix
        # should be ndarray by default
        assert(isinstance(P,np.ndarray))
        # shape
        assert(np.all(P.shape==(self.msm.nstates,self.msm.nstates)))
        # test transition matrix properties
        import pyemma.msm.analysis as msmana
        assert(msmana.is_transition_matrix(P))
        assert(msmana.is_connected(P))
        # REVERSIBLE
        assert(msmana.is_reversible(P))

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def test_active_count_fraction(self):
        # should always be a fraction
        assert(self.msm.active_count_fraction >= 0.0 and self.msm.active_count_fraction<= 1.0)
        # special case for this data set:
        assert(self.msm.active_count_fraction == 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        assert(self.msm.active_state_fraction >= 0.0 and self.msm.active_state_fraction<= 1.0)

    def test_effective_count_matrix(self):
        assert(np.allclose(self.tau * self.msm.effective_count_matrix, self.msm.count_matrix_active))

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def test_statdist(self):
        mu = self.msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert(np.all(mu > 0))
        # should sum to one
        assert(np.abs(np.sum(mu)-1.0) < 1e-10)

    def test_eigenvalues(self):
        ev = self.msm.eigenvalues()
        # stochasticity
        assert(np.max(np.abs(ev)) <= 1)
        # irreducible
        assert(np.max(np.abs(ev[1:])) < 1)
        # ordered
        assert(np.all(np.argsort(np.abs(ev))[::-1] == np.arange(len(ev))))
        # REVERSIBLE:
        assert(np.all(np.isreal(ev)))

    def test_eigenvectors_left(self):
        L = self.msm.eigenvectors_left()
        # shape should be right
        assert(np.all(L.shape == (self.msm.nstates,self.msm.nstates)))
        # first one should be identical to stat.dist
        l1 = L[0,:]
        err = self.msm.stationary_distribution - l1
        assert(np.max(np.abs(err)) < 1e-10)
        # sums should be 1, 0, 0, ...
        assert(np.allclose(np.sum(L[1:,:], axis=1), np.zeros(self.msm.nstates-1)))
        # REVERSIBLE:
        assert(np.all(np.isreal(L)))

    def test_eigenvectors_right(self):
        R = self.msm.eigenvectors_right()
        # shape should be right
        assert(np.all(R.shape == (self.msm.nstates,self.msm.nstates)))
        # should be all ones
        r1 = R[:,0]
        assert(np.allclose(r1, np.ones(self.msm.nstates)))
        # REVERSIBLE:
        assert(np.all(np.isreal(R)))

    def test_eigenvectors_RDL(self):
        R = self.msm.eigenvectors_right()
        D = np.diag(self.msm.eigenvalues())
        L = self.msm.eigenvectors_left()

        # orthogonality constraint
        assert(np.allclose(np.dot(R,L), np.eye(self.msm.nstates)))
        # REVERSIBLE: also true for LR because reversible matrix
        assert(np.allclose(np.dot(L,R), np.eye(self.msm.nstates)))
        # recover transition matrix
        assert(np.allclose(np.dot(R, np.dot(D,L)), self.msm.transition_matrix))

    def test_timescales(self):
        ts = self.msm.timescales()
        # should be all positive
        assert(np.all(ts > 0))
        # should be all real
        assert(np.all(np.isreal(ts)))
        # HERE:
        assert(np.max(np.abs(ts[0:3] - np.array([310.87248087, 8.50933441, 5.09082957]))) < 1e-6)

    # ---------------------------------
    # FIRST PASSAGE PROBLEMS
    # ---------------------------------

    def test_committor(self):
        a = 16
        b = 48
        q_forward = self.msm.committor_forward(a,b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)
        assert (np.all(q_forward[:30] < 0.5))
        assert (np.all(q_forward[40:] > 0.5))
        q_backward = self.msm.committor_backward(a,b)
        assert (q_backward[a] == 1)
        assert (q_backward[b] == 0)
        assert (np.all(q_backward[:30] > 0.5))
        assert (np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        assert (np.allclose(q_forward + q_backward, np.ones(self.msm.nstates)))

    def test_mfpt(self):
        a = 16
        b = 48
        t = self.msm.mfpt(a,b)
        assert(t > 0)
        # HERE:
        assert(np.abs(t - 872.69132618104049) < 1e-6)

    # ---------------------------------
    # PCCA
    # ---------------------------------

    def test_pcca_assignment(self):
        ass = self.msm.pcca_assignments(2)
        # test: number of states
        assert(len(ass) == self.msm.nstates)
        # test: should be 0 or 1
        assert(np.all(ass >= 0))
        assert(np.all(ass <= 1))
        # should be equal (zero variance) within metastable sets
        assert(np.std(ass[:30]) == 0)
        assert(np.std(ass[40:]) == 0)

    def test_pcca_distributions(self):
        pccadist = self.msm.pcca_distributions(2)
        # should be right size
        assert(np.all(pccadist.shape == (2,self.msm.nstates)))
        # should be nonnegative
        assert(np.all(pccadist >= 0))
        # should roughly add up to stationary:
        ds = pccadist[0]+pccadist[1]
        ds /= ds.sum()
        assert(np.max(np.abs(ds - self.msm.stationary_distribution)) < 0.001)

    def test_pcca_memberships(self):
        M = self.msm.pcca_memberships(2)
        # should be right size
        assert(np.all(M.shape == (self.msm.nstates,2)))
        # should be nonnegative
        assert(np.all(M >= 0))
        # should add up to one:
        assert(np.allclose(np.sum(M, axis=1), np.ones(self.msm.nstates)))

    def test_pcca_sets(self):
        S = self.msm.pcca_sets(2)
        assignment = self.msm.pcca_assignments(2)
        # should coincide with assignment
        for i,s in enumerate(S):
            for j in range(len(s)):
                assert(assignment[s[j]] == i)

    # ---------------------------------
    # EXPERIMENTAL STUFF
    # ---------------------------------

    def expectation(self):
        e = self.msm.expectation(range(self.msm.nstates))
        assert(np.abs(e - 31.73) < 0.01)

    def test_correlation(self):
        # raise assertion error because size is wrong:
        a = [1,2,3]
        with self.assertRaises(AssertionError):
            self.msm.correlation(a, 1)
        # should decrease
        a = range(self.msm.nstates)
        corr1 = self.msm.correlation(a, range(1000))
        assert(len(corr1) == 1000)
        assert(corr1[0] > corr1[-1])
        a = range(self.msm.nstates)
        corr2 = self.msm.correlation(a, range(1000), b=a)
        # should be indentical to autocorr
        assert(np.all(corr1 == corr2))
        # Test: should be increasing in time
        b = range(self.msm.nstates)[::-1]
        corr3 = self.msm.correlation(a, range(1000), b)
        assert(len(corr3) == 1000)
        assert(corr3[0] < corr3[-1])

    def test_relaxation(self):
        pi_perturbed = (self.msm.stationary_distribution**2)
        pi_perturbed /= pi_perturbed.sum()
        a = range(self.msm.nstates)
        rel1 = self.msm.relaxation(self.msm.stationary_distribution, a, range(1000))
        # should be constant because we are in equilibrium
        assert(np.allclose(rel1-rel1[0], np.zeros((np.shape(rel1)[0]))))
        rel2 = self.msm.relaxation(pi_perturbed, a, range(1000))
        # should relax
        assert(len(rel2) == 1000)
        assert(rel2[0] < rel2[-1])

    def test_fingerprint_correlation(self):
        # raise assertion error because size is wrong:
        a = [1,2,3]
        with self.assertRaises(AssertionError):
            self.msm.fingerprint_correlation(a, 1)
        # should decrease
        a = range(self.msm.nstates)
        fp1 = self.msm.fingerprint_correlation(a)
        # first timescale is infinite
        assert(fp1[0][0] == np.inf)
        # next timescales are identical to timescales:
        assert(np.allclose(fp1[0][1:], self.msm.timescales()))
        # all amplitudes nonnegative (for autocorrelation)
        assert(np.all(fp1[1][:] >= 0))
        # identical call
        b = range(self.msm.nstates)
        fp2 = self.msm.fingerprint_correlation(a, b)
        assert(np.allclose(fp1[0],fp2[0]))
        assert(np.allclose(fp1[1],fp2[1]))
        # should be - of the above, apart from the first
        b = range(self.msm.nstates)[::-1]
        fp3 = self.msm.fingerprint_correlation(a, b)
        assert(np.allclose(fp1[0],fp3[0]))
        assert(np.allclose(fp1[1][1:],-fp3[1][1:]))

    def test_fingerprint_relaxation(self):
        # raise assertion error because size is wrong:
        a = [1,2,3]
        with self.assertRaises(AssertionError):
            self.msm.fingerprint_relaxation(self.msm.stationary_distribution, a)
        # equilibrium relaxation should be constant
        a = range(self.msm.nstates)
        fp1 = self.msm.fingerprint_relaxation(self.msm.stationary_distribution, a)
        # first timescale is infinite
        assert(fp1[0][0] == np.inf)
        # next timescales are identical to timescales:
        assert(np.allclose(fp1[0][1:], self.msm.timescales()))
        # dynamical amplitudes should be near 0 because we are in equilibrium
        assert(np.max(np.abs(fp1[1][1:])) < 1e-10)
        # off-equilibrium relaxation
        pi_perturbed = (self.msm.stationary_distribution**2)
        pi_perturbed /= pi_perturbed.sum()
        fp2 = self.msm.fingerprint_relaxation(pi_perturbed, a)
        # first timescale is infinite
        assert(fp2[0][0] == np.inf)
        # next timescales are identical to timescales:
        assert(np.allclose(fp2[0][1:], self.msm.timescales()))
        # dynamical amplitudes should be significant because we are not in equilibrium
        assert(np.max(np.abs(fp2[1][1:])) > 0.1)

    # ---------------------------------
    # STATISTICS, SAMPLING
    # ---------------------------------

    def test_active_state_indexes(self):
        I = self.msm.active_state_indexes
        assert(len(I) == self.msm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt
        hist = dt.count_states(self.msm.discrete_trajectories_full)
        # number of frames should match on active subset
        A = self.msm.active_set
        for i in range(A.shape[0]):
            assert(I[i].shape[0] == hist[A[i]])
            assert(I[i].shape[1] == 2)

    def test_generate_traj(self):
        T = 10
        gt = self.msm.generate_traj(T)
        # Test: should have the right dimension
        assert(np.all(gt.shape == (T,2)))
        # itraj should be right
        assert(np.all(gt[:,0] == 0))

    def test_sample_by_state(self):
        nsample = 100
        ss = self.msm.sample_by_state(nsample)
        # must have the right size
        assert(len(ss) == self.msm.nstates)
        # must be correctly assigned
        dtraj_active = self.msm.discrete_trajectories_active[0]
        for i,samples in enumerate(ss):
            # right shape
            assert(np.all(samples.shape == (nsample,2)))
            for row in samples:
                assert(row[0] == 0) # right trajectory
                assert(dtraj_active[row[1]] == i)

    def test_trajectory_weights(self):
        W = self.msm.trajectory_weights()
        # should sum to 1
        assert(np.abs(np.sum(W[0])-1.0) < 1e-6)

    # ----------------------------------
    # MORE COMPLEX TESTS / SANITY CHECKS
    # ----------------------------------

    def test_two_state_kinetics(self):
        # sanity check: k_forward + k_backward = 1.0/t2 for the two-state process
        l2 = self.msm.eigenvectors_left()[1,:]
        core1 = np.argmin(l2)
        core2 = np.argmax(l2)
        # transition time from left to right and vice versa
        t12 = self.msm.mfpt(core1, core2)
        t21 = self.msm.mfpt(core2, core1)
        # relaxation time
        t2 = self.msm.timescales()[0]
        # the following should hold roughly = k12 + k21 = k2.
        # sum of forward/backward rates can be a bit smaller because we are using small cores and therefore underestimate rates
        ksum = 1.0/self.msm.mfpt(core1, core2) + 1.0/self.msm.mfpt(core2, core1)
        k2 = 1.0/self.msm.timescales()[0]
        assert(np.abs(k2 - ksum) < 0.001)

if __name__=="__main__":
    unittest.main()
