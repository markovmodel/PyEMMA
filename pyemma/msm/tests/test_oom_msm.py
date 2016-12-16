
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


r"""Unit test for the OOM-based MSM estimation.

"""

from __future__ import absolute_import
import unittest

import numpy as np
import scipy.sparse
import scipy.linalg as scl
import warnings
import pkg_resources

from pyemma.msm.estimators import OOM_based_MSM
from pyemma.msm import markov_model
from pyemma.util.linalg import _sort_by_norm
import msmtools.estimation as msmest
from six.moves import range

def oom_transformations(Ct, C2t, rank):
    # Number of states:
    N = Ct.shape[0]
    # Get the SVD of Ctau:
    U, s, V = scl.svd(Ct, full_matrices=False)
    # Reduce:
    s = s[:rank]
    U = U[:, :rank]
    V = V[:rank, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s**(-0.5)))
    F2 = np.dot(V, np.diag(s**(-0.5)))
    # Compute observable operators:
    Xi = np.zeros((rank, N, rank))
    for n in range(N):
        Xi[:, n, :] = np.dot(F1.T, np.dot(C2t[:, :, n], F2))
    Xi_full = np.sum(Xi, axis=1)
    # Compute evaluator:
    c = np.sum(Ct, axis=1)
    sigma = np.dot(F1.T, c)
    # Compute information state:
    l, R = scl.eig(Xi_full.T)
    l, R = _sort_by_norm(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l

def TransitionMatrix(Xi, omega, sigma, reversible=True):
    N = Xi.shape[1]
    Ct_Eq = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Ct_Eq[i, j] = np.dot(omega.T, np.dot(Xi[:, i, :], np.dot(Xi[:, j, :], sigma)))
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    pi_r = np.sum(Ct_Eq, axis=1)
    if reversible:
        pi_c = np.sum(Ct_Eq, axis=0)
        pi = pi_r + pi_c
        Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi[:, None]
    else:
        Tt_Eq = Ct_Eq / pi_r[:, None]

    return Tt_Eq


class TestMSMFiveState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data:
        data = np.load(pkg_resources.resource_filename(__name__, "data/TestData_OOM_MSM.npz"))
        cls.dtrajs = [data['arr_%d'%k] for k in range(1000)]

        # Number of states:
        cls.N = 5
        # Lag time:
        cls.tau = 5
        # Rank:
        cls.rank = 3
        # Build models:
        cls.msmrev = OOM_based_MSM(lag=cls.tau)
        cls.msmrev.fit(cls.dtrajs)
        cls.msm = OOM_based_MSM(lag=cls.tau, reversible=False)
        cls.msm.fit(cls.dtrajs)

        """Sparse"""
        cls.msmrev_sparse = OOM_based_MSM(lag=cls.tau, sparse=True)
        cls.msmrev_sparse.fit(cls.dtrajs)
        cls.msm_sparse = OOM_based_MSM(lag=cls.tau, reversible=False, sparse=True)
        cls.msm_sparse.fit(cls.dtrajs)

        # Reference count matrices at lag time tau and 2*tau:
        cls.C2t = data['C2t']
        cls.Ct = np.sum(cls.C2t, axis=1)

        # Compute OOM-components:
        cls.Xi, cls.omega, cls.sigma, cls.l = oom_transformations(cls.Ct, cls.C2t, cls.rank)
        # Compute corrected transition matrix:
        Tt_rev = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=True)
        Tt = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=False)

        # Build reference models:
        cls.rmsmrev = markov_model(Tt_rev)
        cls.rmsm = markov_model(Tt)

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_reversible(self):
        # Reversible
        assert self.msmrev.is_reversible
        assert self.msmrev_sparse.is_reversible
        # Non-reversible
        assert not self.msm.is_reversible
        assert not self.msm_sparse.is_reversible

    def _sparse(self, msm):
        assert not (msm.is_sparse)

    def test_sparse(self):
        self._sparse(self.msmrev_sparse)
        self._sparse(self.msm_sparse)

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msm)
        self._lagtime(self.msmrev_sparse)
        self._lagtime(self.msm_sparse)

    def test_active_set(self):

        assert np.all(self.msmrev.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_sparse.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm.active_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm_sparse.active_set == np.arange(self.N, dtype=int))

    def test_largest_connected_set(self):
        assert np.all(self.msmrev.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msmrev_sparse.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm.largest_connected_set == np.arange(self.N, dtype=int))
        assert np.all(self.msm_sparse.largest_connected_set == np.arange(self.N, dtype=int))

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 5)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msm)
        self._nstates(self.msmrev_sparse)
        self._nstates(self.msm_sparse)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msm)
        self._connected_sets(self.msmrev_sparse)
        self._connected_sets(self.msm_sparse)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msm)
        self._connectivity(self.msmrev_sparse)
        self._connectivity(self.msm_sparse)

    def _count_matrix_active(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_active.toarray()
        else:
            C = msm.count_matrix_active
        assert np.allclose(C, self.Ct)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msm)
        self._count_matrix_active(self.msmrev_sparse, sparse=True)
        self._count_matrix_active(self.msm_sparse, sparse=True)

    def _count_matrix_full(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_full.toarray()
        else:
            C = msm.count_matrix_full
        assert np.allclose(C, self.Ct)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msm)
        self._count_matrix_full(self.msmrev_sparse, sparse=True)
        self._count_matrix_full(self.msm_sparse, sparse=True)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_full[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_full)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msm)
        self._discrete_trajectories_full(self.msmrev_sparse)
        self._discrete_trajectories_full(self.msm_sparse)

    def _discrete_trajectories_active(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_active[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_active)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msm)
        self._discrete_trajectories_active(self.msmrev_sparse)
        self._discrete_trajectories_active(self.msm_sparse)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
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
        # Test equality with model:
        if isinstance(P, scipy.sparse.csr_matrix):
            P = P.toarray()
        if msm.is_reversible:
            assert np.allclose(P, self.rmsmrev.transition_matrix)
        else:
            assert np.allclose(P, self.rmsm.transition_matrix)

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)
        self._transition_matrix(self.msmrev_sparse)
        self._transition_matrix(self.msm_sparse)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)
        # For this data set:
        assert (msm.active_state_fraction == 1.0)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msm)
        self._active_state_fraction(self.msmrev_sparse)
        self._active_state_fraction(self.msm_sparse)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)
        # Should match model:
        if msm.is_reversible:
            assert np.allclose(mu, self.rmsmrev.stationary_distribution)
        else:
            assert np.allclose(mu, self.rmsm.stationary_distribution)


    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msm)
        self._statdist(self.msmrev_sparse)
        self._statdist(self.msm_sparse)

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
        self._eigenvalues(self.msmrev_sparse)
        self._eigenvalues(self.msm_sparse)

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
        self._eigenvectors_left(self.msm)
        self._eigenvectors_left(self.msmrev_sparse)
        self._eigenvectors_left(self.msm_sparse)

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
        self._eigenvectors_right(self.msm)
        self._eigenvectors_right(self.msmrev_sparse)
        self._eigenvectors_right(self.msm_sparse)

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
        self._eigenvectors_RDL(self.msmrev_sparse)
        self._eigenvectors_RDL(self.msm_sparse)

    def _timescales(self, msm):
        if not msm.is_reversible:
            with warnings.catch_warnings(record=True) as w:
                ts = msm.timescales()
        else:
            ts = msm.timescales()

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            ts_ref = self.rmsmrev.timescales()
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau*ts_ref, decimal=2)
        else:
            ts_ref = self.rmsm.timescales()
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau*ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)
        self._timescales(self.msmrev_sparse)
        self._timescales(self.msm_sparse)

    def _eigenvalues_OOM(self, msm):
        assert np.allclose(msm.eigenvalues_OOM, self.l)

    def test_eigenvalues_OOM(self):
        self._eigenvalues_OOM(self.msmrev)
        self._eigenvalues_OOM(self.msm)
        self._eigenvalues_OOM(self.msmrev_sparse)
        self._eigenvalues_OOM(self.msm_sparse)

    def _oom_components(self, msm):
        Xi = msm.OOM_components
        omega = msm.OOM_omega
        sigma = msm.OOM_sigma
        assert np.allclose(Xi, self.Xi)
        assert np.allclose(omega, self.omega)
        assert np.allclose(sigma, self.sigma)

    def test_oom_components(self):
        self._oom_components(self.msmrev)
        self._oom_components(self.msm)
        self._oom_components(self.msmrev_sparse)
        self._oom_components(self.msm_sparse)


class TestMSM_Incomplete(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data:
        data = np.load(pkg_resources.resource_filename(__name__, "data/TestData_OOM_MSM.npz"))
        indices = np.array([21, 25, 30, 40, 66, 72, 74, 91, 116, 158, 171, 175, 201, 239, 246, 280, 300, 301, 310, 318,
                            322, 323, 339, 352, 365, 368, 407, 412, 444, 475, 486, 494, 510, 529, 560, 617, 623, 637,
                            676, 689, 728, 731, 778, 780, 811, 828, 838, 845, 851, 859, 868, 874, 895, 933, 935, 938,
                            958, 961, 968, 974, 984, 990, 999])
        cls.dtrajs = []
        for k in range(1000):
            if k not in indices:
                cls.dtrajs.append(data['arr_%d'%k])

        # Number of states:
        cls.N = 5
        # Lag time:
        cls.tau = 5
        # Rank:
        cls.rank = 2
        # Build models:
        cls.msmrev = OOM_based_MSM(lag=cls.tau)
        cls.msmrev.fit(cls.dtrajs)
        cls.msm = OOM_based_MSM(lag=cls.tau, reversible=False)
        cls.msm.fit(cls.dtrajs)

        """Sparse"""
        cls.msmrev_sparse = OOM_based_MSM(lag=cls.tau, sparse=True)
        cls.msmrev_sparse.fit(cls.dtrajs)
        cls.msm_sparse = OOM_based_MSM(lag=cls.tau, reversible=False, sparse=True)
        cls.msm_sparse.fit(cls.dtrajs)

        # Reference count matrices at lag time tau and 2*tau:
        cls.C2t = data['C2t_s']
        cls.Ct = np.sum(cls.C2t, axis=1)
        # Restrict to active set:
        lcc = msmest.largest_connected_set(cls.Ct)
        cls.Ct_active = msmest.largest_connected_submatrix(cls.Ct, lcc=lcc)
        cls.C2t_active = cls.C2t[:4, :4, :4]
        cls.active_fraction = np.sum(cls.Ct_active) / np.sum(cls.Ct)

        # Compute OOM-components:
        cls.Xi, cls.omega, cls.sigma, cls.l = oom_transformations(cls.Ct_active, cls.C2t_active, cls.rank)
        # Compute corrected transition matrix:
        Tt_rev = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=True)
        Tt = TransitionMatrix(cls.Xi, cls.omega, cls.sigma, reversible=False)

        # Build reference models:
        cls.rmsmrev = markov_model(Tt_rev)
        cls.rmsm = markov_model(Tt)

    # ---------------------------------
    # BASIC PROPERTIES
    # ---------------------------------

    def test_reversible(self):
        # Reversible
        assert self.msmrev.is_reversible
        assert self.msmrev_sparse.is_reversible
        # Non-reversible
        assert not self.msm.is_reversible
        assert not self.msm_sparse.is_reversible

    def _sparse(self, msm):
        assert not (msm.is_sparse)

    def test_sparse(self):
        self._sparse(self.msmrev_sparse)
        self._sparse(self.msm_sparse)

    def _lagtime(self, msm):
        assert (msm.lagtime == self.tau)

    def test_lagtime(self):
        self._lagtime(self.msmrev)
        self._lagtime(self.msm)
        self._lagtime(self.msmrev_sparse)
        self._lagtime(self.msm_sparse)

    def test_active_set(self):

        assert np.all(self.msmrev.active_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msmrev_sparse.active_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msm.active_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msm_sparse.active_set == np.arange(self.N-1, dtype=int))

    def test_largest_connected_set(self):
        assert np.all(self.msmrev.largest_connected_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msmrev_sparse.largest_connected_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msm.largest_connected_set == np.arange(self.N-1, dtype=int))
        assert np.all(self.msm_sparse.largest_connected_set == np.arange(self.N-1, dtype=int))

    def _nstates(self, msm):
        # should always be <= full
        assert (msm.nstates <= msm.nstates_full)
        # THIS DATASET:
        assert (msm.nstates == 4)

    def test_nstates(self):
        self._nstates(self.msmrev)
        self._nstates(self.msm)
        self._nstates(self.msmrev_sparse)
        self._nstates(self.msm_sparse)

    def _connected_sets(self, msm):
        cs = msm.connected_sets
        assert (len(cs) >= 1)
        # MODE LARGEST:
        assert (np.all(cs[0] == msm.active_set))

    def test_connected_sets(self):
        self._connected_sets(self.msmrev)
        self._connected_sets(self.msm)
        self._connected_sets(self.msmrev_sparse)
        self._connected_sets(self.msm_sparse)

    def _connectivity(self, msm):
        # HERE:
        assert (msm.connectivity == 'largest')

    def test_connectivity(self):
        self._connectivity(self.msmrev)
        self._connectivity(self.msm)
        self._connectivity(self.msmrev_sparse)
        self._connectivity(self.msm_sparse)

    def _count_matrix_active(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_active.toarray()
        else:
            C = msm.count_matrix_active
        assert np.allclose(C, self.Ct_active)

    def test_count_matrix_active(self):
        self._count_matrix_active(self.msmrev)
        self._count_matrix_active(self.msm)
        self._count_matrix_active(self.msmrev_sparse, sparse=True)
        self._count_matrix_active(self.msm_sparse, sparse=True)

    def _count_matrix_full(self, msm, sparse=False):
        if sparse:
            C = msm.count_matrix_full.toarray()
        else:
            C = msm.count_matrix_full
        assert np.allclose(C, self.Ct)

    def test_count_matrix_full(self):
        self._count_matrix_full(self.msmrev)
        self._count_matrix_full(self.msm)
        self._count_matrix_full(self.msmrev_sparse, sparse=True)
        self._count_matrix_full(self.msm_sparse, sparse=True)

    def _discrete_trajectories_full(self, msm):
        assert (np.all(self.dtrajs[0] == msm.discrete_trajectories_full[0]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_full)

    def test_discrete_trajectories_full(self):
        self._discrete_trajectories_full(self.msmrev)
        self._discrete_trajectories_full(self.msm)
        self._discrete_trajectories_full(self.msmrev_sparse)
        self._discrete_trajectories_full(self.msm_sparse)

    def _discrete_trajectories_active(self, msm):
        dtraj = self.dtrajs[15]
        dtraj[dtraj==4] = -1
        assert (np.all(dtraj == msm.discrete_trajectories_active[15]))
        assert len(self.dtrajs) == len(msm.discrete_trajectories_active)

    def test_discrete_trajectories_active(self):
        self._discrete_trajectories_active(self.msmrev)
        self._discrete_trajectories_active(self.msm)
        self._discrete_trajectories_active(self.msmrev_sparse)
        self._discrete_trajectories_active(self.msm_sparse)

    def _transition_matrix(self, msm):
        P = msm.transition_matrix
        # should be ndarray by default
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
        # Test equality with model:
        if isinstance(P, scipy.sparse.csr_matrix):
            P = P.toarray()
        if msm.is_reversible:
            assert np.allclose(P, self.rmsmrev.transition_matrix)
        else:
            assert np.allclose(P, self.rmsm.transition_matrix)

    def test_transition_matrix(self):
        self._transition_matrix(self.msmrev)
        self._transition_matrix(self.msm)
        self._transition_matrix(self.msmrev_sparse)
        self._transition_matrix(self.msm_sparse)

    # ---------------------------------
    # SIMPLE STATISTICS
    # ---------------------------------

    def _active_state_fraction(self, msm):
        # should always be a fraction
        assert (0.0 <= msm.active_state_fraction <= 1.0)
        # For this data set:
        assert (msm.active_state_fraction == 0.8)

    def test_active_state_fraction(self):
        # should always be a fraction
        self._active_state_fraction(self.msmrev)
        self._active_state_fraction(self.msm)
        self._active_state_fraction(self.msmrev_sparse)
        self._active_state_fraction(self.msm_sparse)

    # ---------------------------------
    # EIGENVALUES, EIGENVECTORS
    # ---------------------------------

    def _statdist(self, msm):
        mu = msm.stationary_distribution
        # should strictly positive (irreversibility)
        assert (np.all(mu > 0))
        # should sum to one
        assert (np.abs(np.sum(mu) - 1.0) < 1e-10)
        # Should match model:
        if msm.is_reversible:
            assert np.allclose(mu, self.rmsmrev.stationary_distribution)
        else:
            assert np.allclose(mu, self.rmsm.stationary_distribution)


    def test_statdist(self):
        self._statdist(self.msmrev)
        self._statdist(self.msm)
        self._statdist(self.msmrev_sparse)
        self._statdist(self.msm_sparse)

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
        self._eigenvalues(self.msmrev_sparse)
        self._eigenvalues(self.msm_sparse)

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
        self._eigenvectors_left(self.msm)
        self._eigenvectors_left(self.msmrev_sparse)
        self._eigenvectors_left(self.msm_sparse)

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
        self._eigenvectors_right(self.msm)
        self._eigenvectors_right(self.msmrev_sparse)
        self._eigenvectors_right(self.msm_sparse)

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
        self._eigenvectors_RDL(self.msmrev_sparse)
        self._eigenvectors_RDL(self.msm_sparse)

    def _timescales(self, msm):
        if not msm.is_reversible:
            with warnings.catch_warnings(record=True) as w:
                ts = msm.timescales()
        else:
            ts = msm.timescales()

        # should be all positive
        assert (np.all(ts > 0))
        # REVERSIBLE: should be all real
        if msm.is_reversible:
            ts_ref = self.rmsmrev.timescales()
            assert (np.all(np.isreal(ts)))
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau*ts_ref, decimal=2)
        else:
            ts_ref = self.rmsm.timescales()
            # HERE:
            np.testing.assert_almost_equal(ts, self.tau*ts_ref, decimal=2)

    def test_timescales(self):
        self._timescales(self.msmrev)
        self._timescales(self.msm)
        self._timescales(self.msmrev_sparse)
        self._timescales(self.msm_sparse)

    def _eigenvalues_OOM(self, msm):
        assert np.allclose(msm.eigenvalues_OOM, self.l)

    def test_eigenvalues_OOM(self):
        self._eigenvalues_OOM(self.msmrev)
        self._eigenvalues_OOM(self.msm)
        self._eigenvalues_OOM(self.msmrev_sparse)
        self._eigenvalues_OOM(self.msm_sparse)

    def _oom_components(self, msm):
        Xi = msm.OOM_components
        omega = msm.OOM_omega
        sigma = msm.OOM_sigma
        assert np.allclose(Xi, self.Xi)
        assert np.allclose(omega, self.omega)
        assert np.allclose(sigma, self.sigma)

    def test_oom_components(self):
        self._oom_components(self.msmrev)
        self._oom_components(self.msm)
        self._oom_components(self.msmrev_sparse)
        self._oom_components(self.msm_sparse)

if __name__ == "__main__":
    unittest.main()