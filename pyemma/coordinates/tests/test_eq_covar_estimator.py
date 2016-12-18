import numpy as np
import scipy.linalg as scl
import unittest
import pkg_resources

from pyemma.coordinates import covariance_lagged
from pyemma.coordinates.estimation.koopman import _KoopmanEstimator
from pyemma.coordinates import source


class TestEqCovar(unittest.TestCase):
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
        cls.source_obj = source(cls.data, chunk_size=200)

        # Lag time:
        cls.tau = 10
        # Truncation for small eigenvalues:
        cls.epsilon = 1e-6
        # Chunksize:
        cls.chunksize = 200

        # Set constant mean for tests:
        cls.mean_constant = np.random.rand(cls.nf)

        # Generate _KoopmanEstimator:
        Kest = _KoopmanEstimator(cls.tau, epsilon=cls.epsilon, chunksize=cls.chunksize)
        Kest.estimate(cls.source_obj)
        cls.weight_object = Kest.weights

        # References for xx=True
        cls.mx = np.zeros(cls.nf)
        cls.my = np.zeros(cls.nf)
        cls.mx_c = np.zeros(cls.nf)
        cls.my_c = np.zeros(cls.nf)
        cls.Mxx = np.zeros((cls.nf, cls.nf))
        cls.Mxy = np.zeros((cls.nf, cls.nf))
        cls.Mxx_c = np.zeros((cls.nf, cls.nf))
        cls.Mxy_c = np.zeros((cls.nf, cls.nf))
        cls.Mxx_sym = np.zeros((cls.nf, cls.nf))
        cls.Mxy_sym = np.zeros((cls.nf, cls.nf))
        cls.Mxx_c_sym = np.zeros((cls.nf, cls.nf))
        cls.Mxy_c_sym = np.zeros((cls.nf, cls.nf))
        cls.Mxx0 = np.zeros((cls.nf, cls.nf))
        cls.Mxy0 = np.zeros((cls.nf, cls.nf))
        cls.Mxx0_sym = np.zeros((cls.nf, cls.nf))
        cls.Mxy0_sym = np.zeros((cls.nf, cls.nf))
        cls.wt = 0
        it = cls.source_obj.iterator(lag=cls.tau, return_trajindex=False)
        # Computations with data mean:
        for X, Y in it:
            w = cls.weight_object.weights(X)
            Xc = (X - cls.mean_constant[None, :]).copy()
            Yc = (Y - cls.mean_constant[None, :]).copy()
            cls.mx += np.sum(w[:, None] * X, axis=0)
            cls.my += np.sum(w[:, None] * Y, axis=0)
            cls.mx_c += np.sum(w[:, None] * Xc, axis=0)
            cls.my_c += np.sum(w[:, None] * Yc, axis=0)
            cls.Mxx += np.dot((w[:, None]*X).T, X)
            cls.Mxy += np.dot((w[:, None]*X).T, Y)
            cls.Mxx_c += np.dot((w[:, None] * Xc).T, Xc)
            cls.Mxy_c += np.dot((w[:, None] * Xc).T, Yc)
            cls.Mxx_sym += np.dot((w[:, None]*X).T, X) + np.dot((w[:, None]*Y).T, Y)
            cls.Mxy_sym += np.dot((w[:, None]*X).T, Y) + np.dot((w[:, None]*Y).T, X)
            cls.Mxx_c_sym += np.dot((w[:, None]*Xc).T, Xc) + np.dot((w[:, None]*Yc).T, Yc)
            cls.Mxy_c_sym += np.dot((w[:, None]*Xc).T, Yc) + np.dot((w[:, None]*Yc).T, Xc)
            cls.wt += w.sum()
        cls.mx /= cls.wt
        cls.my /= cls.wt
        cls.msym = 0.5*(cls.mx + cls.my)
        cls.mx_c /= cls.wt
        cls.my_c /= cls.wt
        cls.msym_c = 0.5*(cls.mx_c + cls.my_c)
        cls.Mxx /= cls.wt
        cls.Mxy /= cls.wt
        cls.Mxx_c /= cls.wt
        cls.Mxy_c /= cls.wt
        cls.Mxx_sym /= 2*cls.wt
        cls.Mxy_sym /= 2*cls.wt
        cls.Mxx_c_sym /= 2*cls.wt
        cls.Mxy_c_sym /= 2*cls.wt

        # Computations without data mean:
        it = cls.source_obj.iterator(lag=cls.tau, return_trajindex=False)
        for X, Y in it:
            w = cls.weight_object.weights(X)
            X0 = (X - cls.mx[None, :]).copy()
            Y0 = (Y - cls.my[None, :]).copy()
            X0_sym = (X - cls.msym[None, :]).copy()
            Y0_sym = (Y - cls.msym[None, :]).copy()
            cls.Mxx0 += np.dot((w[:, None]*X0).T, X0)
            cls.Mxy0 += np.dot((w[:, None]*X0).T, Y0)
            cls.Mxx0_sym += np.dot((w[:, None]*X0_sym).T, X0_sym) + np.dot((w[:, None]*Y0_sym).T, Y0_sym)
            cls.Mxy0_sym += np.dot((w[:, None]*X0_sym).T, Y0_sym) + np.dot((w[:, None]*Y0_sym).T, X0_sym)
        cls.Mxx0 /= cls.wt
        cls.Mxy0 /= cls.wt
        cls.Mxx0_sym /= 2*cls.wt
        cls.Mxy0_sym /= 2*cls.wt

    def test_XX(self):
        cc = covariance_lagged(data=self.data, c0t=False, lag=self.tau, bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, c0t=False, lag=self.tau, bessel=False, weights=self.weight_object)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.cov, self.Mxx)
        assert np.allclose(cc1.mean, self.mx)
        assert np.allclose(cc1.cov, self.Mxx)

    def test_XX_removeconstantmean(self):
        cc = covariance_lagged(data=self.data, c0t=False, lag=self.tau, remove_constant_mean=self.mean_constant,
                               bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, c0t=False, lag=self.tau, remove_constant_mean=self.mean_constant,
                                bessel=False, weights=self.weight_object)
        assert np.allclose(cc.mean, self.mx_c)
        assert np.allclose(cc.cov, self.Mxx_c)
        assert np.allclose(cc1.mean, self.mx_c)
        assert np.allclose(cc1.cov, self.Mxx_c)

    def test_XX_removedatamean(self):
        cc = covariance_lagged(data=self.data, c0t=False, lag=self.tau, remove_data_mean=True, bessel=False,
                               weights="koopman")
        cc1 = covariance_lagged(data=self.data, c0t=False, lag=self.tau, remove_data_mean=True, bessel=False,
                                weights=self.weight_object)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.cov, self.Mxx0)
        assert np.allclose(cc1.mean, self.mx)
        assert np.allclose(cc1.cov, self.Mxx0)

    def test_XY(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, bessel=False, weights=self.weight_object)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.mean_tau, self.my)
        assert np.allclose(cc.cov, self.Mxx)
        assert np.allclose(cc.cov_tau, self.Mxy)
        assert np.allclose(cc1.mean, self.mx)
        assert np.allclose(cc1.mean_tau, self.my)
        assert np.allclose(cc1.cov, self.Mxx)
        assert np.allclose(cc1.cov_tau, self.Mxy)


    def test_XY_removeconstantmean(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, remove_constant_mean=self.mean_constant,
                               bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, remove_constant_mean=self.mean_constant,
                                bessel=False, weights="koopman")
        assert np.allclose(cc.mean, self.mx_c)
        assert np.allclose(cc.mean_tau, self.my_c)
        assert np.allclose(cc.cov, self.Mxx_c)
        assert np.allclose(cc.cov_tau, self.Mxy_c)
        assert np.allclose(cc1.mean, self.mx_c)
        assert np.allclose(cc1.mean_tau, self.my_c)
        assert np.allclose(cc1.cov, self.Mxx_c)
        assert np.allclose(cc1.cov_tau, self.Mxy_c)

    def test_XY_removedatamean(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, remove_data_mean=True, bessel=False,
                               weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, remove_data_mean=True, bessel=False,
                                weights=self.weight_object)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.mean_tau, self.my)
        assert np.allclose(cc.cov, self.Mxx0)
        assert np.allclose(cc.cov_tau, self.Mxy0)
        assert np.allclose(cc1.mean, self.mx)
        assert np.allclose(cc1.mean_tau, self.my)
        assert np.allclose(cc1.cov, self.Mxx0)
        assert np.allclose(cc1.cov_tau, self.Mxy0)

    def test_XY_sym(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True, bessel=False,
                               weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True, bessel=False,
                                weights=self.weight_object)
        assert np.allclose(cc.mean, self.msym)
        assert np.allclose(cc.cov, self.Mxx_sym)
        assert np.allclose(cc.cov_tau, self.Mxy_sym)
        assert np.allclose(cc1.mean, self.msym)
        assert np.allclose(cc1.cov, self.Mxx_sym)
        assert np.allclose(cc1.cov_tau, self.Mxy_sym)

    def test_XY_sym_removeconstantmean(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True,
                               remove_constant_mean=self.mean_constant, bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True,
                                remove_constant_mean=self.mean_constant, bessel=False, weights=self.weight_object)
        assert np.allclose(cc.mean, self.msym_c)
        assert np.allclose(cc.cov, self.Mxx_c_sym)
        assert np.allclose(cc.cov_tau, self.Mxy_c_sym)
        assert np.allclose(cc1.mean, self.msym_c)
        assert np.allclose(cc1.cov, self.Mxx_c_sym)
        assert np.allclose(cc1.cov_tau, self.Mxy_c_sym)

    def test_XY_sym_removedatamean(self):
        cc = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True, remove_data_mean=True,
                               bessel=False, weights="koopman")
        cc1 = covariance_lagged(data=self.data, lag=self.tau, c0t=True, reversible=True, remove_data_mean=True,
                                bessel=False, weights=self.weight_object)
        assert np.allclose(cc.mean, self.msym)
        assert np.allclose(cc.cov, self.Mxx0_sym)
        assert np.allclose(cc.cov_tau, self.Mxy0_sym)
        assert np.allclose(cc1.mean, self.msym)
        assert np.allclose(cc1.cov, self.Mxx0_sym)
        assert np.allclose(cc1.cov_tau, self.Mxy0_sym)




if __name__ == "__main__":
    unittest.main()