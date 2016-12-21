import numpy as np
import scipy.linalg as scl
import unittest
import pkg_resources

from pyemma._ext.variational.solvers.direct import sort_by_norm
from pyemma.coordinates.estimation.koopman import _KoopmanEstimator
from pyemma.coordinates import source


class TestKoopman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Basis set definition:
        cls.nf = 10
        cls.chi = np.zeros((20, cls.nf), dtype=float)
        for n in range(cls.nf):
            cls.chi[2*n:2*(n+1), n] = 1.0

        # Load simulations:
        f = np.load(pkg_resources.resource_filename(__name__, "data/test_data_koopman.npz"))
        trajs = [f[key] for key in f.keys()]
        cls.data = [cls.chi[traj, :] for traj in trajs]
        cls.source_obj = source(cls.data)

        # Lag time:
        cls.tau = 10
        # Truncation for small eigenvalues:
        cls.epsilon = 1e-6
        # Chunksize:
        cls.chunksize = 200

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
        cls.C0 *= (1.0 / cls.frames)
        cls.Ct *= (1.0 / cls.frames)

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
        cls.u_pc_1 = np.real(Un[:, 0])
        v = np.eye(cls.N1, 1, k=-cls.N1+1)[:, 0]
        cls.u_pc_1 *= (1.0 / np.dot(cls.u_pc_1, v))

        # Transform u to original basis:
        cls.u = np.zeros(cls.nf + 1)
        cls.u[:cls.nf] = np.dot(cls.R, cls.u_pc_1[:-1])
        cls.u[cls.nf] = cls.u_pc_1[-1] - np.dot(np.dot(cls.mean_x, cls.R), cls.u_pc_1[:-1])

        # Set up the model:
        cls.K_est = _KoopmanEstimator(cls.tau, epsilon=cls.epsilon, chunksize=cls.chunksize)
        cls.K_est.estimate(cls.source_obj)

    def test_K_pc_1(self):
        assert np.allclose(self.K_est.K_pc_1, self.K)

    def test_u_pc_1(self):
        assert np.allclose(self.K_est.u_pc_1, self.u_pc_1)

    def test_u(self):
        assert np.allclose(self.K_est.u, self.u)

    def test_weights(self):
        # Apply weights to original first traj:
        sample_traj = np.dot(self.data[0], self.u[:-1]) + self.u[-1] * np.ones((self.data[0].shape[0], 1))
        weight_object = self.K_est.weights
        assert np.allclose(weight_object.weights(self.data[0]), sample_traj)

    def test_R(self):
        assert np.allclose(self.K_est.R, self.R)

    def test_mean(self):
        assert np.allclose(self.K_est.mean, self.mean_x)

if __name__ == "__main__":
    unittest.main()