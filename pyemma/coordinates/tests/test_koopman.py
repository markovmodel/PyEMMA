import numpy as np
import scipy.linalg as scl
import unittest

import pyemma.coordinates as pco

from variational.solvers.direct import sort_by_norm

class TestKoopman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Basis set definition:
        cls.nf = 10
        cls.chi = np.zeros((20, cls.nf), dtype=float)
        for n in range(cls.nf):
            cls.chi[2*n:2*(n+1), n] = 1.0

        # Load simulations:
        f = np.load("data/test_data_koopman.npz")
        trajs = [f[key] for key in f.keys()]
        cls.data = [cls.chi[traj, :] for traj in trajs]

        # Lag time:
        cls.tau = 10
        # Truncation for small eigenvalues:
        cls.epsilon = 1e-6

        # Compute the means:
        cls.mean_x = np.zeros(cls.nf)
        cls.mean_y = np.zeros(cls.nf)
        cls.frames = 0
        for traj in cls.data:
            cls.mean_x += np.sum(traj[:-cls.tau, :], axis=0)
            cls.mean_y += np.sum(traj[cls.tau:, :], axis=0)
            cls.frames += traj[:-cls.tau, :].shape[0]
        cls.mean_x *= (1.0 / cls.frames)
        cls.mean_y *= (1.0 / cls.frames)

        # Compute correlations:
        cls.C0 = np.zeros((cls.nf, cls.nf))
        cls.Ct = np.zeros((cls.nf, cls.nf))
        for traj in cls.data:
            itraj = (traj - cls.mean_x[None, :]).copy()
            cls.C0 += np.dot(itraj[:-cls.tau, :].T, itraj[:-cls.tau, :])
            cls.Ct += np.dot(itraj[:-cls.tau, :].T, itraj[cls.tau:, :])
        cls.C0 *= (1.0 / (cls.frames-1))
        cls.Ct *= (1.0 / (cls.frames-1))

        # Compute whitening transformation:
        d, V = scl.eigh(cls.C0)
        evmin = np.minimum(0, np.min(d))
        ep = np.maximum(-evmin, cls.epsilon)
        d, V = sort_by_norm(d, V)
        ind = np.where(np.abs(d) > ep)[0]
        d = d[ind]
        V = V[:, ind]
        for j in range(V.shape[1]):
            jj = np.argmax(np.abs(V[:, j]))
            V[:, j] *= np.sign(V[jj, j])
        cls.R = np.dot(V, np.diag(d**(-0.5)))

        # Compute non-reversible Koopman matrix:
        cls.K = np.dot(cls.R.T, np.dot(cls.Ct, cls.R))
        cls.K = np.vstack((cls.K, np.dot((cls.mean_y - cls.mean_x), cls.R)))
        cls.K = np.hstack((cls.K, np.eye(cls.K.shape[0], 1, k=-cls.K.shape[0]+1)))
        cls.N1 = cls.K.shape[0]

        # Compute its eigenvalues:
        cls.ln, cls.Rn = scl.eig(cls.K)
        cls.ln, cls.Rn = sort_by_norm(cls.ln, cls.Rn)

        # Compute u-vector:
        ln, Un = scl.eig(cls.K.T)
        ln, Un = sort_by_norm(ln, Un)
        cls.u = np.real(Un[:, 0])
        v = np.eye(cls.N1, 1, k=-cls.N1+1)[:, 0]
        cls.u *= (1.0 / np.dot(cls.u, v))

        # Compute reversible C0:
        cls.C0_eq = np.zeros((cls.N1, cls.N1))
        for traj in cls.data:
            traj = np.dot((traj - cls.mean_x[None, :]), cls.R).copy()
            traj = np.hstack((traj, np.ones((traj.shape[0], 1))))
            w = np.dot(traj, cls.u)
            cls.C0_eq += np.dot((w[:-cls.tau, None] * traj[:-cls.tau, :]).T, traj[:-cls.tau, :])
        cls.C0_eq *= (1.0 / (cls.frames-1))

        # Compute equilibrium means:
        cls.mean_eq = cls.C0_eq[:, -1]

        # Compute reversible Ct:
        cls.Ct_eq = 0.5*(np.dot(cls.C0_eq, cls.K) + np.dot(cls.K.T, cls.C0_eq))

        # Compute reversible whitening transformation:
        d, V = scl.eigh(cls.C0_eq)
        evmin = np.minimum(0, np.min(d))
        ep = np.maximum(-evmin, cls.epsilon)
        d, V = sort_by_norm(d, V)
        ind = np.where(np.abs(d) > ep)[0]
        d = d[ind]
        V = V[:, ind]
        for j in range(V.shape[1]):
            jj = np.argmax(np.abs(V[:, j]))
            V[:, j] *= np.sign(V[jj, j])
        cls.R_eq = np.dot(V, np.diag(d**(-0.5)))

        # Compute reversible K:
        cls.K_eq = np.dot(cls.R_eq.T, np.dot(cls.Ct_eq, cls.R_eq))

        # Compute its eigenvalues:
        cls.lr, cls.Rr = scl.eigh(cls.K_eq)
        cls.lr, cls.Rr = sort_by_norm(cls.lr, cls.Rr)

        # Set up the model:
        cls.koop = pco.koopman(data=cls.data, lag=cls.tau)
        cls.koop_eq = pco.koopman(data=cls.data, lag=cls.tau, eq=True)

    def test_mean_x(self):
        assert np.allclose(self.koop.mean, self.mean_x)
        assert np.allclose(self.koop_eq.mean_pc, self.mean_eq)

    def test_C0(self):
        assert np.allclose(self.koop.cov, self.C0)
        assert np.allclose(self.koop_eq.cov_pc, self.C0_eq)

    def test_Ct(self):
        assert np.allclose(self.koop.cov_tau, self.Ct)
        assert np.allclose(self.koop_eq.cov_tau_pc, self.Ct_eq)

    def test_K(self):
        assert np.allclose(self.koop.koopman_matrix, self.K)
        assert np.allclose(self.koop_eq.koopman_matrix, self.K_eq)

    def test_u(self):
        assert np.allclose(self.koop_eq._model.u, self.u)

    def test_eigenvalues(self):
        assert np.allclose(self.koop.eigenvalues, self.ln)
        assert np.allclose(self.koop_eq.eigenvalues, self.lr)


if __name__ == "__main__":
    unittest.main()