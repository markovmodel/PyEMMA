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
        self.dtrajs = generate_traj(P, 10000, start=0)
        self.tau = 1

        """Estimate MSM"""
        self.C_MSM = cmatrix(self.dtrajs, self.tau, sliding=True)
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
        msm = markov_state_model(self.dtrajs, self.tau)
        assert_allclose(self.dtrajs, msm.discrete_trajectories)
        self.assertEqual(self.tau, msm.lagtime)
        assert_allclose(self.lcc_MSM, msm.largest_connected_set)
        self.assertTrue(allclose_sparse(self.Ccc_MSM, msm.count_matrix_active))
        self.assertTrue(allclose_sparse(self.C_MSM, msm.count_matrix_full))
        self.assertTrue(allclose_sparse(self.P_MSM, msm.transition_matrix))
        assert_allclose(self.mu_MSM, msm.stationary_distribution)
        assert_allclose(self.ts, msm.timescales(self.k))

class TestMSMDoubleWellReversible(unittest.TestCase):

    def setUp(self):
        testpath = abspath(join(abspath(__file__), pardir)) + '../../util/tests/data/'
        import pyemma.util.discrete_trajectories as dt
        self.dtraj = dt.read_discrete_trajectory(testpath+'2well_traj_100K.dat')
        self.tau = 10
        self.msm = markov_state_model(self.dtraj, self.tau)

    # compute, computed
    def test_compute(self):
        # should give warning
        with warnings.catch_warnings(record=True) as w:
            self.msm.compute()

    def test_computed(self):
        assert(self.msm.computed)

    # BASIC PROPERTIES

    def test_active_set(self):
        # should always be <= full set
        assert(len(self.msm.active_set) <= self.msm._n_full)
        # REVERSIBLE: should be range(msm.n_states)
        assert(len(self.msm.active_set) == self.msm.nstates)
        assert(np.all(self.msm.active_set == range(self.msm.nstates)))

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
        assert(np.all(self.dtraj, self.msm.discrete_trajectories_full))

    def test_discrete_trajectories_active(self):
        dta = self.msm.discrete_trajectories_active
        # HERE
        assert(len(dta) == 1)
        # HERE: states are shifted down from the beginning, because early states are missing
        assert(dta[0][0] < self.dtraj[0])

    # DERIVED QUANTITIES
    def expectation(self):
        e = self.msm.expectation(range(self.msm.nstates))
        assert(np.abs(e - 31.73) < 0.01)

    def test_correlation(self):
        # raise assertion error because size is wrong:
        a = [1,2,3]
        with AssertionError:
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
        assert(corr1[0] < corr1[-1])

    def test_relaxation(self):
        pi_perturbed = (self.msm.stationary_distribution**2)
        pi_perturbed /= pi_perturbed.sum()
        a = range(self.msm.nstates)
        rel1 = self.msm.relaxation(self.msm.stationary_distribution, a, range(1000))
        # should be zero because we are in equilibrium
        assert(np.allclose(rel1, np.zeros((np.shape(rel1)[0]))))
        rel2 = self.msm.relaxation(pi_perturbed, a, range(1000))
        # should relax
        assert(len(rel2) == 1000)
        assert(rel2[0] < rel2[-1])

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


    def test_committor(self):
        a = 16
        b = 48
        q_forward = self.msm.committor_forward(a,b)
        assert (q_forward[a] == 0)
        assert (q_forward[b] == 1)
        assert (np.all(q_forward[:30] < 0.5))
        assert (np.all(q_forward[40:] > 0.5))
        q_backward = self.msm.committor_backward(a,b)
        assert (q_forward[a] == 1)
        assert (q_forward[b] == 0)
        assert (np.all(q_backward[:30] > 0.5))
        assert (np.all(q_backward[40:] < 0.5))
        # REVERSIBLE:
        assert (np.allclose(np.abs(q_forward + q_backward, np.ones(self.msm.nstates))))

    def test_active_state_indexes(self):
        I = self.msm.active_state_indexes
        assert(len(I) == self.msm.nstates)
        # compare to histogram
        import pyemma.util.discrete_trajectories as dt
        hist = dt.count_states(self.msm.discrete_trajectories_full)
        # number of frames should match on active subset
        for (i,s) in self.msm.active_set:
            assert(I[i].shape[0] == hist[s])
            assert(I[i].shape[1] == 2)


if __name__=="__main__":
    unittest.main()
