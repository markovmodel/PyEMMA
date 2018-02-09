# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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


"""
@author: paul
"""

from __future__ import absolute_import
import unittest
import numpy as np
from pyemma.coordinates import vamp as pyemma_api_vamp
from pyemma.msm import estimate_markov_model
from logging import getLogger

from pyemma.msm.estimators._dtraj_stats import cvsplit_dtrajs

logger = getLogger('pyemma.'+'TestVAMP')


def random_matrix(n, rank=None, eps=0.01):
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    if rank is None:
        rank = n
    if rank > n:
        rank = n
    s = np.concatenate((np.maximum(s, eps)[0:rank], np.zeros(n-rank)))
    return u.dot(np.diag(s)).dot(v)


def _check_serialize(vamp):
    import six
    if six.PY2:
        return vamp
    import tempfile
    import pyemma
    try:
        with tempfile.NamedTemporaryFile(delete=False) as ntf:
            vamp.save(ntf.name)
            restored = pyemma.load(ntf.name)

        np.testing.assert_allclose(restored.model.C00, vamp.model.C00)
        np.testing.assert_allclose(restored.model.C0t, vamp.model.C0t)
        np.testing.assert_allclose(restored.model.Ctt, vamp.model.Ctt)
        np.testing.assert_allclose(restored.cumvar, vamp.cumvar)
        assert_allclose_ignore_phase(restored.singular_values, vamp.singular_values)
        assert_allclose_ignore_phase(restored.singular_vectors_left, vamp.singular_vectors_left)
        assert_allclose_ignore_phase(restored.singular_vectors_right, vamp.singular_vectors_right)
        np.testing.assert_equal(restored.dimension(), vamp.dimension())
        return restored
    finally:
        import os
        os.remove(ntf.name)


class TestVAMPEstimatorSelfConsistency(unittest.TestCase):
    def test_full_rank(self):
        self.do_test(20, 20, test_partial_fit=True)

    def test_low_rank(self):
        dim = 30
        rank = 15
        self.do_test(dim, rank, test_partial_fit=True)

    def do_test(self, dim, rank, test_partial_fit=False):
        # setup
        N_frames = [123, 456, 789]
        N_trajs = len(N_frames)
        A = random_matrix(dim, rank)
        trajs = []
        mean = np.random.randn(dim)
        for i in range(N_trajs):
            # set up data
            white = np.random.randn(N_frames[i], dim)
            brown = np.cumsum(white, axis=0)
            correlated = np.dot(brown, A)
            trajs.append(correlated + mean)

        # test
        tau = 50
        vamp = pyemma_api_vamp(trajs, lag=tau, scaling=None)
        vamp.right = True
        _check_serialize(vamp)

        assert vamp.dimension() <= rank

        atol = np.finfo(vamp.output_type()).eps*10.0
        rtol = np.finfo(vamp.output_type()).resolution
        phi_trajs = [ sf[tau:, :] for sf in vamp.get_output() ]
        phi = np.concatenate(phi_trajs)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.dimension()), rtol=rtol, atol=atol)

        vamp.right = False
        psi_trajs = [ sf[0:-tau, :] for sf in vamp.get_output() ]
        psi = np.concatenate(psi_trajs)
        mean_left = psi.sum(axis=0) / psi.shape[0]
        cov_left = psi.T.dot(psi) / psi.shape[0]
        np.testing.assert_allclose(mean_left, 0.0, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cov_left, np.eye(vamp.dimension()), rtol=rtol, atol=atol)

        # compute correlation between left and right
        assert phi.shape[0]==psi.shape[0]
        C01_psi_phi = psi.T.dot(phi) / phi.shape[0]
        n = max(C01_psi_phi.shape)
        C01_psi_phi = C01_psi_phi[0:n,:][:, 0:n]
        np.testing.assert_allclose(C01_psi_phi, np.diag(vamp.singular_values[0:vamp.dimension()]), rtol=rtol, atol=atol)

        if test_partial_fit:
            vamp2 = pyemma_api_vamp(lag=tau, scaling=None)
            for t in trajs:
                vamp2.partial_fit(t)
                vamp2 = _check_serialize(vamp2)

            model_params = vamp._model.get_model_params()
            model_params2 = vamp2._model.get_model_params()

            atol = 1e-14
            rtol = 1e-5

            for n in model_params.keys():
                if model_params[n] is not None and model_params2[n] is not None:
                    if n not in ('U', 'V'):
                        np.testing.assert_allclose(model_params[n], model_params2[n], rtol=rtol, atol=atol,
                                               err_msg='failed for model param %s' % n)
                    else:
                        assert_allclose_ignore_phase(model_params[n], model_params2[n], atol=atol)

            vamp2.singular_values # trigger diagonalization

            vamp2.right = True
            for t, ref in zip(trajs, phi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[tau:]), ref, rtol=rtol, atol=atol)

            vamp2.right = False
            for t, ref in zip(trajs, psi_trajs):
                assert_allclose_ignore_phase(vamp2.transform(t[0:-tau]), ref, rtol=rtol, atol=atol)


def generate(T, N_steps, s0=0):
    dtraj = np.zeros(N_steps, dtype=int)
    s = s0
    T_cdf = T.cumsum(axis=1)
    for t in range(N_steps):
        dtraj[t] = s
        s = np.searchsorted(T_cdf[s, :], np.random.rand())
    return dtraj


def assert_allclose_ignore_phase(A, B, atol=1e-14, rtol=1e-5):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    assert A.shape == B.shape
    for i in range(B.shape[1]):
        assert (np.allclose(A[:, i], B[:, i], atol=atol, rtol=rtol)
                or np.allclose(A[:, i], -B[:, i], atol=atol, rtol=rtol))


class TestVAMPModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N_steps = 10000
        N_traj = 20
        lag = 1
        T = np.linalg.matrix_power(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), lag)
        dtrajs = [generate(T, N_steps) for _ in range(N_traj)]
        p0 = np.zeros(3)
        p1 = np.zeros(3)
        trajs = []
        for dtraj in dtrajs:
            traj = np.zeros((N_steps, T.shape[0]))
            traj[np.arange(len(dtraj)), dtraj] = 1.0
            trajs.append(traj)
            p0 += traj[:-lag, :].sum(axis=0)
            p1 += traj[lag:, :].sum(axis=0)
        vamp = pyemma_api_vamp(trajs, lag=lag, scaling=None, dim=1.0)
        msm = estimate_markov_model(dtrajs, lag=lag, reversible=False)
        cls.trajs = trajs
        cls.dtrajs = dtrajs
        cls.lag = lag
        cls.msm = msm
        cls.vamp = vamp
        cls.p0 = p0 / p0.sum()
        cls.p1 = p1 / p1.sum()
        cls.atol = np.finfo(vamp.output_type()).eps*1000.0

    def test_K_is_T(self):
        m0 = self.vamp.model.mean_0
        mt = self.vamp.model.mean_t
        C0 = self.vamp.model.C00 + m0[:, np.newaxis]*m0[np.newaxis, :]
        C1 = self.vamp.model.C0t + m0[:, np.newaxis]*mt[np.newaxis, :]
        K = np.linalg.inv(C0).dot(C1)
        np.testing.assert_allclose(K, self.msm.P, atol=1E-5)

        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.P).dot(np.diag(self.p1 ** -0.5))
        np.testing.assert_allclose(np.linalg.svd(Tsym)[1][1:], self.vamp.singular_values[0:2], atol=1E-7)

    def test_singular_functions_against_MSM(self):
        Tsym = np.diag(self.p0 ** 0.5).dot(self.msm.P).dot(np.diag(self.p1 ** -0.5))
        Up, S, Vhp = np.linalg.svd(Tsym)
        Vp = Vhp.T
        U = Up * (self.p0 ** -0.5)[:, np.newaxis]
        V = Vp * (self.p1 ** -0.5)[:, np.newaxis]
        assert_allclose_ignore_phase(U[:, 0], np.ones(3), atol=1E-5)
        assert_allclose_ignore_phase(V[:, 0], np.ones(3), atol=1E-5)
        U = U[:, 1:]
        V = V[:, 1:]
        self.vamp.right = True
        phi = self.vamp.transform(np.eye(3))
        self.vamp.right = False
        psi = self.vamp.transform(np.eye(3))
        assert_allclose_ignore_phase(U, psi, atol=1E-5)
        assert_allclose_ignore_phase(V, phi, atol=1E-5)
        references_sf = [U.T.dot(np.diag(self.p0)).dot(np.linalg.matrix_power(self.msm.P, k*self.lag)).dot(V).T for k in
                         range(10-1)]
        cktest = self.vamp.cktest(n_observables=2, mlags=10)
        pred_sf = cktest.predictions
        esti_sf = cktest.estimates
        for e, p, r in zip(esti_sf[1:], pred_sf[1:], references_sf[1:]):
            np.testing.assert_allclose(np.diag(p), np.diag(r), atol=1E-6)
            np.testing.assert_allclose(np.abs(p), np.abs(r), atol=1E-6)

    def test_CK_expectation_against_MSM(self):
        obs = np.eye(3) # observe every state
        cktest = self.vamp.cktest(observables=obs, statistics=None, mlags=4)
        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]

        for i, (est_, pred_) in enumerate(zip(est, pred)):
            msm = estimate_markov_model(dtrajs=self.dtrajs, lag=self.lag*(i+1), reversible=False)
            msm_esti = self.p0.T.dot(msm.P).dot(obs)
            msm_pred = self.p0.T.dot(np.linalg.matrix_power(self.msm.P, (i+1))).dot(obs)
            np.testing.assert_allclose(pred_,  msm_pred, atol=self.atol)
            np.testing.assert_allclose(est_, msm_esti, atol=self.atol)
            np.testing.assert_allclose(est_, pred_, atol=0.006)

    def test_CK_covariances_of_singular_functions(self):
        cktest = self.vamp.cktest(n_observables=2, mlags=4)  # auto
        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]
        error = np.max(np.abs(np.array(pred) - np.array(est))) / max(np.max(pred), np.max(est))
        assert error < 0.05

    def test_CK_covariances_against_MSM(self):
        obs = np.eye(3) # observe every state
        sta = np.eye(3) # restrict p0 to every state
        cktest = self.vamp.cktest(observables=obs, statistics=sta, mlags=4, show_progress=True)
        pred = cktest.predictions[1:]
        est = cktest.estimates[1:]

        for i, (est_, pred_) in enumerate(zip(est, pred)):
            msm = estimate_markov_model(dtrajs=self.dtrajs, lag=self.lag*(i+1), reversible=False)
            msm_esti = (self.p0 * sta).T.dot(msm.P).dot(obs).T
            msm_pred = (self.p0 * sta).T.dot(np.linalg.matrix_power(self.msm.P, (i+1))).dot(obs).T
            np.testing.assert_allclose(np.diag(pred_),  np.diag(msm_pred), atol=self.atol)
            np.testing.assert_allclose(np.diag(est_), np.diag(msm_esti), atol=self.atol)
            np.testing.assert_allclose(np.diag(est_), np.diag(pred_), atol=0.006)

    def test_self_score_with_MSM(self):
        T = self.msm.P
        Tadj = np.diag(1./self.p1).dot(T.T).dot(np.diag(self.p0))
        NFro = np.trace(T.dot(Tadj))
        s2 = self.vamp.score(score_method='VAMP2')
        np.testing.assert_allclose(s2, NFro)

        Tsym = np.diag(self.p0**0.5).dot(T).dot(np.diag(self.p1**-0.5))
        Nnuc = np.linalg.norm(Tsym, ord='nuc')
        s1 = self.vamp.score(score_method='VAMP1')
        np.testing.assert_allclose(s1, Nnuc)

        # TODO: check why this is not equal
        sE = self.vamp.score(score_method='VAMPE')
        np.testing.assert_allclose(sE, NFro)  # see paper appendix H.2

    def test_score_vs_MSM(self):
        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(32):
            trajs_test, trajs_train = cvsplit_dtrajs(self.trajs)
        with numpy_random_seed(32):
            dtrajs_test, dtrajs_train = cvsplit_dtrajs(self.dtrajs)

        methods = ('VAMP1', 'VAMP2', 'VAMPE')

        for m in methods:
            msm_train = estimate_markov_model(dtrajs=dtrajs_train, lag=self.lag, reversible=False)
            score_msm = msm_train.score(dtrajs_test, score_method=m, score_k=None)

            vamp_train = pyemma_api_vamp(data=trajs_train, lag=self.lag, dim=1.0)
            score_vamp = vamp_train.score(test_data=trajs_test, score_method=m)

            self.assertAlmostEqual(score_msm, score_vamp, places=2 if m == 'VAMPE' else 3, msg=m)

    def test_kinetic_map(self):
        lag = 10
        self.vamp = pyemma_api_vamp(self.trajs, lag=lag, scaling='km', right=False)
        transformed = [t[:-lag] for t in self.vamp.get_output()]
        std = np.std(np.concatenate(transformed), axis=0)
        np.testing.assert_allclose(std, self.vamp.singular_values[:self.vamp.dimension()], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
