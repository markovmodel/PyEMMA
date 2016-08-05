# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from __future__ import absolute_import
import unittest
from six.moves import range

import numpy as np
import pyemma.thermo
from pyemma.thermo import EmptyState
import warnings
import msmtools

def tower_sample(distribution):
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, rnd)

def generate_trajectory(transition_matrices, bias_energies, K, n_samples, x0):
    """generates a list of TRAM trajs"""

    ttraj = np.ones(n_samples, np.intc)*K
    dtraj = np.zeros(n_samples, dtype=np.intc)
    btraj = np.zeros((n_samples, bias_energies.shape[0]), dtype=np.float64)

    x = x0
    dtraj[0] = x
    btraj[0, :] = bias_energies[:, x]
    h = 1
    for s in range(n_samples-1):
        x_new = tower_sample(transition_matrices[K, x, :])
        x = x_new
        dtraj[h] = x
        btraj[h, :] = bias_energies[:, x]
        h += 1
    return (ttraj, dtraj, btraj)

def generate_simple_trajectory(transition_matrix, n_samples, x0):
    """generates a list of TRAM trajs"""

    n_states = transition_matrix.shape[0]
    traj = np.zeros(n_samples, dtype=int)
    x = x0
    traj[0] = x
    h = 1
    for s in range(n_samples-1):
        x_new = tower_sample(transition_matrix[x,:])
        assert x_new < n_states
        x = x_new
        traj[h] = x
        h += 1
    return traj

def global_equilibrium_samples(pi, n_samples):
    samples = np.zeros(n_samples, dtype=int)
    for s in range(n_samples):
        samples[s] = tower_sample(pi)
    return samples

def T_matrix(energy):
    n = energy.shape[0]
    metropolis = energy[np.newaxis, :] - energy[:, np.newaxis]
    metropolis[(metropolis < 0.0)] = 0.0
    selection = np.zeros((n,n))
    selection += np.diag(np.ones(n-1)*0.5,k=1)
    selection += np.diag(np.ones(n-1)*0.5,k=-1)
    selection[0,0] = 0.5
    selection[-1,-1] = 0.5
    metr_hast = selection * np.exp(-metropolis)
    for i in range(metr_hast.shape[0]):
        metr_hast[i, i] = 0.0
        metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
    return metr_hast


class TestTRAMexceptions(unittest.TestCase):
    def test_warning_empty_ensemble(self):
        # have no samples in ensemble #0
        bias_energies = np.zeros((2, 2))
        bias_energies[1, :] = np.array([0.0, 0.0])
        T = np.zeros((2, 2, 2))
        T[1, :, :] = T_matrix(bias_energies[1,:])
        n_samples = 100
        trajs = generate_trajectory(T, bias_energies, 1, n_samples, 0)
        trajs = ([trajs[0]], [trajs[1]], [trajs[2]])
        tram = pyemma.thermo.TRAM(lag=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tram.estimate(trajs)
            assert len(w) >= 1
            assert any(issubclass(v.category, EmptyState) for v in w)

    def test_exception_wrong_format(self):
        btraj = np.zeros((10,3))
        ttraj = 4*np.ones(10, dtype=int)
        dtraj = np.ones(10, dtype=int)
        tram = pyemma.thermo.TRAM(lag=1)
        with self.assertRaises(AssertionError):
            tram.estimate(([ttraj], [dtraj], [btraj]))


class TestTRAMwith5StateDTRAMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bias_energies = np.zeros((2,5))
        cls.bias_energies[0,:] = np.array([1.0,0.0,10.0,0.0,1.0])
        cls.bias_energies[1,:] = np.array([100.0,0.0,0.0,0.0,100.0])
        cls.T = np.zeros((2,5,5))
        cls.T[0,:,:] = T_matrix(cls.bias_energies[0,:])
        cls.T[1,:,:] = T_matrix(cls.bias_energies[1,:])

        n_samples = 50000

        cls.bias_energies_sh = cls.bias_energies - cls.bias_energies[0,:]
        data = []
        data.append(generate_trajectory(cls.T, cls.bias_energies_sh, 0, n_samples, 0))
        data.append(generate_trajectory(cls.T, cls.bias_energies_sh, 0, n_samples, 4))
        data.append(generate_trajectory(cls.T, cls.bias_energies_sh, 1, n_samples, 2))
        cls.trajs = tuple(list(x) for x in zip(*data)) # "transpose" list of tuples to a tuple of lists

    def test_5_state_model(self):
        self.run_5_state_model(False)

    def test_5_state_model_direct(self):
        self.run_5_state_model(True)

    def run_5_state_model(self, direct_space):
        tram = pyemma.thermo.TRAM(lag=1, maxerr=1E-12, save_convergence_info=10, direct_space=direct_space, nn=1, init='mbar')
        tram.estimate(self.trajs)

        log_pi_K_i = tram.biased_conf_energies.copy()
        log_pi_K_i[0,:] -= np.min(log_pi_K_i[0,:])
        log_pi_K_i[1,:] -= np.min(log_pi_K_i[1,:])
        assert np.allclose(log_pi_K_i, self.bias_energies, atol=0.1)

        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.log_likelihood()+1.E-5>=tram.loglikelihoods[0:-1])

        # simple test: just call the methods
        tram.pointwise_free_energies()
        tram.mbar_pointwise_free_energies()


class TestTRAMasReversibleMSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_states = 50
        traj_length = 10000

        dtraj = np.zeros(traj_length, dtype=int)
        dtraj[::2] = np.random.randint(1, n_states, size=(traj_length-1)//2+1)

        c = msmtools.estimation.count_matrix(dtraj, lag=1)
        while not msmtools.estimation.is_connected(c, directed=True):
            dtraj = np.zeros(traj_length, dtype=int)
            dtraj[::2] = np.random.randint(1, n_states, size=(traj_length-1)//2+1)
            c = msmtools.estimation.count_matrix(dtraj, lag=1)

        #state_counts = np.bincount(dtraj)[:,np.newaxis]
        ttraj = np.zeros(traj_length, dtype=int)
        btraj = np.zeros((traj_length,1))
        cls.tram_trajs = ([ttraj], [dtraj], [btraj])

        cls.T_ref = msmtools.estimation.tmatrix(c, reversible=True).toarray()
        
    def test_reversible_msm(self):
        self.reversible_msm(False)

    def test_reversible_msm_direct(self):
        self.reversible_msm(True)

    def reversible_msm(self, direct_space):
        tram = pyemma.thermo.TRAM(lag=1, maxerr=1.E-12, save_convergence_info=10, direct_space=direct_space, nn=None)
        tram.estimate(self.tram_trajs)
        assert np.allclose(self.T_ref,  tram.models[0].transition_matrix, atol=1.E-4)

        # Lagrange multipliers should be > 0
        assert np.all(tram.log_lagrangian_mult > -1.E300)
        # lower bound on the log-likelihood must be maximal at convergence
        assert np.all(tram.log_likelihood()+1.E-5 >= tram.loglikelihoods[0:-1])


class TRAMandTRAMMBARBaseClass(object):
    @classmethod
    def setUpClass(cls, init_trammbar):
        # (1-D FEL) TRAM unit test
        # (1) define mu(k,x) on a fine grid, k=0 is defined as unbiased
        # (2) define coarse grid (of Markov states) on x
        # (3) -> compute pi from the grid definition and mu. Compute the conditional mu_i^k
        # (4) -> from pi, generate transtion matrix
        # (5) -> run two-level stochastic process to generate the bias trajectories
        # (6) optional if init_trammbar=True, add a global equilibrium trajectory for TRAMMBAR

        cls.test_trammbar = init_trammbar

        # (1)
        if init_trammbar:
            therm_states_local = [0, 1, 3] # 2 has only equilibrium data
            therm_states_global = [1, 2] # 1 has both kinds of data
        else:
            therm_states_local = [0, 1, 2]
            therm_states_global = []
        therm_states = list(set(therm_states_local).union(therm_states_global))
        n_therm_states_local = len(therm_states_local)
        n_therm_states_global = len(therm_states_global)
        n_therm_states_total = max(therm_states)+1
        n_conf_states = 3
        n_micro_states = 50
        traj_length = 30000

        mu = np.zeros((n_therm_states_total, n_micro_states))
        for k in therm_states:
            mu[k, :] = np.random.rand(n_micro_states)*0.8 + 0.2
            if k>0:
               mu[k,:] *= (np.random.rand()*0.8 + 0.2)
        energy = -np.log(mu)
        # (2)
        chi = np.zeros((n_micro_states, n_conf_states)) # (crisp)
        for i in range(n_conf_states):
            chi[n_micro_states*i//n_conf_states:n_micro_states*(i+1)//n_conf_states, i] = 1
        assert np.allclose(chi.sum(axis=1), np.ones(n_micro_states))
        # (3)
        #             k  x  i                 k           x  i
        mu_joint = mu[:, :, np.newaxis] * chi[np.newaxis, :, :]
        assert np.allclose(mu_joint.sum(axis=2), mu)
        z = mu_joint.sum(axis=1)
        pi = z / z.sum(axis=1)[:, np.newaxis]
        assert np.allclose(pi.sum(axis=1), np.ones(n_therm_states_total))
        mu_conditional = mu_joint / z[:, np.newaxis, :]
        assert np.allclose(mu_conditional.sum(axis=1), np.ones((n_therm_states_total, n_conf_states)))
        # (4)
        T = np.zeros((n_therm_states_total, n_conf_states, n_conf_states))
        for k in therm_states:
            T[k,:,:] = T_matrix(-np.log(pi[k,:]))
            assert np.allclose(T[k,:,:].sum(axis=1), np.ones(n_conf_states))
        # (5)
        ttrajs = []
        dtrajs = []
        btrajs = []
        xes = np.zeros(n_therm_states_local*traj_length, dtype=int)
        C = np.zeros((max(therm_states_local)+1, n_conf_states, n_conf_states), dtype=int)
        h = 0
        for k in therm_states_local:
            ttrajs.append( k*np.ones(traj_length, dtype=int) )
            dtrajs.append( generate_simple_trajectory(T[k, :, :], traj_length, 0) )
            C[k,:,:] = msmtools.estimation.count_matrix(dtrajs[-1], lag=1).toarray()
            btraj = np.zeros((traj_length, n_therm_states_total))
            btrajs.append(btraj)
            for t,i in enumerate(dtrajs[-1]):
                x = tower_sample(mu_conditional[k, :, i])
                assert mu_conditional[k, x, i] > 0
                xes[h*traj_length + t] = x
                btraj[t, :] = energy[:, x] - energy[0, x] # define k=0 as "unbiased"
            h += 1

        # (6)
        if init_trammbar:
            eq_ttrajs = []
            eq_dtrajs = []
            eq_btrajs = []
            eq_xes = np.zeros(n_therm_states_global * traj_length, dtype=int)
            h = 0
            for k in therm_states_global:
                eq_ttrajs.append( k * np.ones(traj_length, dtype=int) )
                eq_dtrajs.append( global_equilibrium_samples(pi[k, :], traj_length) )
                eq_btraj = np.zeros((traj_length, n_therm_states_total))
                eq_btrajs.append(eq_btraj)
                for t, i in enumerate(eq_dtrajs[-1]):
                    x = tower_sample(mu_conditional[k, :, i])
                    assert mu_conditional[k, x, i] > 0
                    eq_xes[h * traj_length + t] = x
                    eq_btraj[t, :] = energy[:, x] - energy[0, x]
                h += 1

        cls.n_conf_states = n_conf_states
        cls.therm_states = therm_states
        cls.therm_states_global = therm_states_global
        cls.therm_states_local = therm_states_local
        cls.n_therm_states_local = n_therm_states_local
        cls.n_therm_states_global = n_therm_states_global
        cls.n_micro_states = n_micro_states
        cls.ttrajs = ttrajs
        cls.dtrajs = dtrajs
        cls.btrajs = btrajs
        if not init_trammbar:
            cls.eq = None
        else:
            cls.eq = [False]*n_therm_states_local + [True]*n_therm_states_global
            cls.eq_ttrajs = eq_ttrajs
            cls.eq_dtrajs = eq_dtrajs
            cls.eq_btrajs = eq_btrajs
            cls.eq_xes = eq_xes
        cls.z = z
        cls.T = T
        cls.C = C
        cls.energy = energy
        cls.mu = mu
        cls.xes = xes

    def with_TRAM_model(self, direct_space):
        # run TRAM
        tram = pyemma.thermo.TRAM(lag=1, maxerr=1E-12, save_convergence_info=10, direct_space=direct_space, nn=None, init='mbar', equilibrium=self.eq)
        if not self.test_trammbar:
            tram.estimate((self.ttrajs, self.dtrajs, self.btrajs))
        else:
            tram.estimate((self.ttrajs + self.eq_ttrajs, self.dtrajs + self.eq_dtrajs, self.btrajs + self.eq_btrajs))

        # csets must include all states
        for k in self.therm_states:
            assert len(tram.csets[k]) == self.n_conf_states

        # check exact identities
        # (1) sum_j v_j T_ji + v_i = sum_j c_ij + sum_j c_ji
        for k in self.therm_states_local:
            lagrangian_mult = np.exp(tram.log_lagrangian_mult[k,:])
            transition_matrix = tram.models[k].transition_matrix
            assert np.allclose(
                lagrangian_mult.T.dot(transition_matrix) + lagrangian_mult,
                self.C[k,:,:].sum(axis=0) + self.C[k,:,:].sum(axis=1))
        total = np.zeros(self.n_conf_states)
        # (2) for TRAM: sum_jk v^k_j T^k_ji = sum_jk c^k_ji
        # (2') for TRAMMBAR: sum_jk v^k_j T^k_ji + sum_k N^k,eq pi_i^k = sum_jk c^k_ji + N_i^eq
        # TODO: include overcounting factor
        pi_biased = np.exp(-tram.biased_conf_energies + tram.therm_energies[:, np.newaxis])
        for k in self.therm_states:
            lagrangian_mult = np.exp(tram.log_lagrangian_mult[k,:])
            transition_matrix = tram.models[k].transition_matrix
            if transition_matrix.shape[0] > 1: # skip equilibrium-only thermodynamic states
                total += lagrangian_mult.T.dot(transition_matrix)
            if self.test_trammbar:
                total += tram.equilibrium_state_counts[k,:].sum()*pi_biased[k,:]
        if not self.test_trammbar:
            assert np.allclose(total, self.C.sum(axis=0).sum(axis=0))
        else:
            assert np.allclose(total, self.C.sum(axis=0).sum(axis=0) + tram.equilibrium_state_counts.sum(axis=0))

        # check transition matrices
        for k in self.therm_states_local:
            assert np.allclose(tram.models[k].transition_matrix, self.T[k,:,:], atol=0.1)

        # check pi
        z_normed = self.z / self.z[0,:].sum()
        assert np.allclose(tram.biased_conf_energies, -np.log(z_normed), atol=0.1)
        pi = np.exp(-tram.biased_conf_energies[0,:])
        pi /= pi.copy().sum()
        assert np.allclose(tram.stationary_distribution, pi) # self-consistency of TRAM

        # check log-likelihood
        assert np.all(tram.log_likelihood()+1.E-5 >= tram.loglikelihoods[0:-1])

        # check mu
        for k in self.therm_states:
            # reference
            f0 = -np.log(self.mu[k, :].sum())
            reference_fel = self.energy[k, :] - f0
            # TRAM result
            test_p_f_es = np.concatenate(tram.pointwise_free_energies(k))
            if not self.test_trammbar:
                counts,_ = np.histogram(self.xes, weights=np.exp(-test_p_f_es), bins=self.n_micro_states)
            else:
                counts, _ = np.histogram(np.concatenate((self.xes, self.eq_xes)), weights=np.exp(-test_p_f_es), bins=self.n_micro_states)
            test_fel = -np.log(counts) + np.log(counts.sum())
            assert np.allclose(reference_fel, test_fel, atol=0.1)


class TestTRAMwithTRAMmodel(unittest.TestCase, TRAMandTRAMMBARBaseClass):
    @classmethod
    def setUpClass(cls):
        TRAMandTRAMMBARBaseClass.setUpClass(False)

    def test_with_TRAM_model_direct(self):
        self.with_TRAM_model(True)

    def test_with_TRAM_model_log_space(self):
        self.with_TRAM_model(False)


class TestTRAMMBARwithTRAMMBARmodel(unittest.TestCase, TRAMandTRAMMBARBaseClass):
    @classmethod
    def setUpClass(cls):
        TRAMandTRAMMBARBaseClass.setUpClass(True)

    def test_TRAMMBAR_log_space(self):
        self.with_TRAM_model(False)

    def test_TRAMMBAR_direct_space(self):
        self.with_TRAM_model(True)


if __name__ == "__main__":
    unittest.main()
