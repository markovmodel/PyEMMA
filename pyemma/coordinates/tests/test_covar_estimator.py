from __future__ import absolute_import
import unittest
import numpy as np

from pyemma.coordinates import covariance_lagged
from pyemma.coordinates import source
from pyemma.coordinates.estimation.koopman import _Weights


__author__ = 'noe'

class weight_object(_Weights):
    def __init__(self):
        self.A = np.random.rand(2)
    def weights(self, X):
        return np.dot(X, self.A)


class TestCovarEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lag = 10
        cls.data = np.random.rand(5000, 2)
        cls.X = cls.data[:-cls.lag, :]
        cls.Y = cls.data[cls.lag:, :]
        cls.T = cls.X.shape[0]
        # Generate iterable
        cls.source_obj = source(cls.data)
        # Chunk size:
        cls.L = 1000
        # Number of chunks:
        cls.nchunks = 10
        # Weights:
        cls.wobj = weight_object()
        # Constant mean to be removed:
        cls.mean_const = np.random.rand(2)
        # Chunksize:
        cls.chunksize = 500

        # moments of X and Y
        cls.w = np.shape(cls.X)[0]
        cls.w_lag0 = np.shape(cls.data)[0]
        cls.wsym = 2*np.shape(cls.X)[0]
        cls.wsym_lag0 = 2*np.shape(cls.data)[0]
        cls.sx = cls.X.sum(axis=0)
        cls.sy = cls.Y.sum(axis=0)
        cls.sx_lag0 = cls.data.sum(axis=0)
        cls.Mxx = (1.0 / cls.w) * np.dot(cls.X.T, cls.X)
        cls.Mxx_lag0 = (1.0 / cls.w_lag0) * np.dot(cls.data.T, cls.data)
        cls.Mxy = (1.0 / cls.w) * np.dot(cls.X.T, cls.Y)
        cls.mx = cls.sx / float(cls.w)
        cls.mx_lag0 = cls.sx_lag0 / float(cls.w_lag0)
        cls.my = cls.sy / float(cls.w)
        cls.X0 = cls.X - cls.mx
        cls.X0_lag0 = cls.data - cls.mx_lag0
        cls.Y0 = cls.Y - cls.my
        cls.Mxx0 = (1.0 / cls.w) * np.dot(cls.X0.T, cls.X0)
        cls.Mxx0_lag0 = (1.0 / cls.w_lag0) * np.dot(cls.X0_lag0.T, cls.X0_lag0)
        cls.Mxy0 = (1.0 / cls.w) * np.dot(cls.X0.T, cls.Y0)

        # moments of x and y, constant mean:
        cls.Xc = cls.X - cls.mean_const
        cls.Xc_lag0 = cls.data - cls.mean_const
        cls.Yc = cls.Y - cls.mean_const
        cls.sx_c = np.sum(cls.Xc, axis=0)
        cls.sx_c_lag0 = np.sum(cls.Xc_lag0, axis=0)
        cls.sy_c = np.sum(cls.Yc, axis=0)
        cls.mx_c = cls.sx_c / float(cls.w)
        cls.mx_c_lag0 = cls.sx_c_lag0 / float(cls.w_lag0)
        cls.my_c = cls.sy_c / float(cls.w)
        cls.Mxx_c = (1.0 / cls.w) * np.dot(cls.Xc.T, cls.Xc)
        cls.Mxx_c_lag0 = (1.0 / cls.w_lag0) * np.dot(cls.Xc_lag0.T, cls.Xc_lag0)
        cls.Mxy_c = (1.0 / cls.w) * np.dot(cls.Xc.T, cls.Yc)

        # symmetric moments
        cls.s_sym = cls.sx + cls.sy
        cls.Mxx_sym = (1.0 / cls.wsym) * (np.dot(cls.X.T, cls.X) + np.dot(cls.Y.T, cls.Y))
        cls.Mxy_sym = (1.0 / cls.wsym) * (np.dot(cls.X.T, cls.Y) + np.dot(cls.Y.T, cls.X))
        cls.m_sym = cls.s_sym / float(cls.wsym)
        cls.X0_sym = cls.X - cls.m_sym
        cls.Y0_sym = cls.Y - cls.m_sym
        cls.Mxx0_sym = (1.0 / cls.wsym) * (np.dot(cls.X0_sym.T, cls.X0_sym) + np.dot(cls.Y0_sym.T, cls.Y0_sym))
        cls.Mxy0_sym = (1.0 / cls.wsym) * (np.dot(cls.X0_sym.T, cls.Y0_sym) + np.dot(cls.Y0_sym.T, cls.X0_sym))

        # symmetric moments, constant mean
        cls.s_c_sym = cls.sx_c + cls.sy_c
        cls.m_c_sym = cls.s_c_sym / float(cls.wsym)
        cls.Mxx_c_sym = (1.0 / cls.wsym) * (np.dot(cls.Xc.T, cls.Xc) + np.dot(cls.Yc.T, cls.Yc))
        cls.Mxy_c_sym = (1.0 / cls.wsym) * (np.dot(cls.Xc.T, cls.Yc) + np.dot(cls.Yc.T, cls.Xc))

        # weighted moments, object case:
        cls.weights_obj = cls.wobj.weights(cls.X)
        cls.weights_obj_lag0 = cls.wobj.weights(cls.data)
        cls.wesum_obj = np.sum(cls.weights_obj)
        cls.wesum_obj_sym = 2*np.sum(cls.weights_obj)
        cls.wesum_obj_lag0 = np.sum(cls.weights_obj_lag0)
        cls.sx_wobj = (cls.weights_obj[:, None] * cls.X).sum(axis=0)
        cls.sx_wobj_lag0 = (cls.weights_obj_lag0[:, None] * cls.data).sum(axis=0)
        cls.sy_wobj = (cls.weights_obj[:, None] * cls.Y).sum(axis=0)
        cls.Mxx_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.X).T, cls.X)
        cls.Mxx_wobj_lag0 = (1.0 / cls.wesum_obj_lag0) * np.dot((cls.weights_obj_lag0[:, None] * cls.data).T, cls.data)
        cls.Mxy_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.X).T, cls.Y)
        cls.mx_wobj = cls.sx_wobj / float(cls.wesum_obj)
        cls.mx_wobj_lag0 = cls.sx_wobj_lag0 / float(cls.wesum_obj_lag0)
        cls.my_wobj = cls.sy_wobj / float(cls.wesum_obj)
        cls.X0_wobj = cls.X - cls.mx_wobj
        cls.X0_wobj_lag0 = cls.data - cls.mx_wobj_lag0
        cls.Y0_wobj = cls.Y - cls.my_wobj
        cls.Mxx0_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.X0_wobj).T, cls.X0_wobj)
        cls.Mxx0_wobj_lag0 = (1.0 / cls.wesum_obj_lag0) * np.dot((cls.weights_obj_lag0[:, None] * cls.X0_wobj_lag0).T
                                                                 , cls.X0_wobj_lag0)
        cls.Mxy0_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.X0_wobj).T, cls.Y0_wobj)

        # weighted symmetric moments, object case:
        cls.s_sym_wobj = cls.sx_wobj + cls.sy_wobj
        cls.Mxx_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] * cls.X).T, cls.X)\
                           + np.dot((cls.weights_obj[:, None] * cls.Y).T, cls.Y))
        cls.Mxy_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] * cls.X).T, cls.Y)\
                           + np.dot((cls.weights_obj[:, None] * cls.Y).T, cls.X))
        cls.m_sym_wobj = cls.s_sym_wobj / float(2 * cls.wesum_obj)
        cls.X0_sym_wobj = cls.X - cls.m_sym_wobj
        cls.Y0_sym_wobj = cls.Y - cls.m_sym_wobj
        cls.Mxx0_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] *cls.X0_sym_wobj).T,cls.X0_sym_wobj)\
                            + np.dot((cls.weights_obj[:, None] *cls.Y0_sym_wobj).T, cls.Y0_sym_wobj))
        cls.Mxy0_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] *cls.X0_sym_wobj).T, cls.Y0_sym_wobj)\
                            + np.dot((cls.weights_obj[:, None] *cls.Y0_sym_wobj).T, cls.X0_sym_wobj))

        # weighted moments, object case, constant mean
        cls.sx_c_wobj = (cls.weights_obj[:, None] * cls.Xc).sum(axis=0)
        cls.sx_c_wobj_lag0 = (cls.weights_obj_lag0[:, None] * cls.Xc_lag0).sum(axis=0)
        cls.sy_c_wobj = (cls.weights_obj[:, None] * cls.Yc).sum(axis=0)
        cls.Mxx_c_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.Xc).T, cls.Xc)
        cls.Mxx_c_wobj_lag0 = (1.0 / cls.wesum_obj_lag0) * np.dot((cls.weights_obj_lag0[:, None] * cls.Xc_lag0).T,
                                                                  cls.Xc_lag0)
        cls.Mxy_c_wobj = (1.0 / cls.wesum_obj) * np.dot((cls.weights_obj[:, None] * cls.Xc).T, cls.Yc)
        cls.mx_c_wobj = cls.sx_c_wobj / float(cls.wesum_obj)
        cls.mx_c_wobj_lag0 = cls.sx_c_wobj_lag0 / float(cls.wesum_obj_lag0)
        cls.my_c_wobj = cls.sy_c_wobj / float(cls.wesum_obj)

        # weighted symmetric moments, object case:
        cls.s_c_sym_wobj = cls.sx_c_wobj + cls.sy_c_wobj
        cls.m_c_sym_wobj = cls.s_c_sym_wobj / float(cls.wesum_obj_sym)
        cls.Mxx_c_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] * cls.Xc).T, cls.Xc)\
                           + np.dot((cls.weights_obj[:, None] * cls.Yc).T, cls.Yc))
        cls.Mxy_c_sym_wobj = (1.0 / cls.wesum_obj_sym) * (np.dot((cls.weights_obj[:, None] * cls.Xc).T, cls.Yc)\
                           + np.dot((cls.weights_obj[:, None] * cls.Yc).T, cls.Xc))

        return cls

    def test_XX_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, c0t=False, remove_data_mean=False, bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_lag0)
        assert np.allclose(cc.cov, self.Mxx_lag0)

    def test_XX_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, c0t=False, remove_data_mean=True, bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_lag0)
        assert np.allclose(cc.cov, self.Mxx0_lag0)

    def test_XX_weightobj_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, c0t=False, remove_data_mean=False, weights=self.wobj, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_wobj_lag0)
        assert np.allclose(cc.cov, self.Mxx_wobj_lag0)

    def test_XX_weightobj_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, c0t=False, remove_data_mean=True, weights=self.wobj, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_wobj_lag0)
        assert np.allclose(cc.cov, self.Mxx0_wobj_lag0)

    def test_XXXY_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=False, c0t=True, lag=self.lag, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.mean_tau, self.my)
        assert np.allclose(cc.cov, self.Mxx)
        assert np.allclose(cc.cov_tau, self.Mxy)

    def test_XXXY_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=True, c0t=True, lag=self.lag, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx)
        assert np.allclose(cc.mean_tau, self.my)
        assert np.allclose(cc.cov, self.Mxx0)
        assert np.allclose(cc.cov_tau, self.Mxy0)

    def test_XXXY_weightobj_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=False, c0t=True, lag=self.lag, weights=self.wobj,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_wobj)
        assert np.allclose(cc.mean_tau, self.my_wobj)
        assert np.allclose(cc.cov, self.Mxx_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy_wobj)

    def test_XXXY_weightobj_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=True, c0t=True, lag=self.lag, weights=self.wobj,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_wobj)
        assert np.allclose(cc.mean_tau, self.my_wobj)
        assert np.allclose(cc.cov, self.Mxx0_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy0_wobj)

    def test_XXXY_sym_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=False, c0t=True, lag=self.lag, reversible=True,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_sym)
        assert np.allclose(cc.cov, self.Mxx_sym)
        assert np.allclose(cc.cov_tau, self.Mxy_sym)

    def test_XXXY_sym_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=True, c0t=True, lag=self.lag, reversible=True,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_sym)
        assert np.allclose(cc.cov, self.Mxx0_sym)
        assert np.allclose(cc.cov_tau, self.Mxy0_sym)

    def test_XXXY_weightobj_sym_withmean(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=False, c0t=True, lag=self.lag, reversible=True,
                               bessel=False, weights=self.wobj, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_sym_wobj)
        assert np.allclose(cc.cov, self.Mxx_sym_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy_sym_wobj)

    def test_XXXY_weightobj_sym_meanfree(self):
        # many passes
        cc = covariance_lagged(data=self.data, remove_data_mean=True, c0t=True, lag=self.lag, reversible=True,
                               bessel=False, weights=self.wobj, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_sym_wobj)
        assert np.allclose(cc.cov, self.Mxx0_sym_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy0_sym_wobj)

    def test_XX_meanconst(self):
        cc = covariance_lagged(data=self.data, c0t=False, remove_constant_mean=self.mean_const, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_c_lag0)
        assert np.allclose(cc.cov, self.Mxx_c_lag0)

    def test_XX_weighted_meanconst(self):
        cc = covariance_lagged(data=self.data, c0t=False, remove_constant_mean=self.mean_const, weights=self.wobj, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_c_wobj_lag0)
        assert np.allclose(cc.cov, self.Mxx_c_wobj_lag0)

    def test_XY_meanconst(self):
        cc = covariance_lagged(data=self.data, remove_constant_mean=self.mean_const, c0t=True, lag=self.lag, bessel=False,
                               chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_c)
        assert np.allclose(cc.mean_tau, self.my_c)
        assert np.allclose(cc.cov, self.Mxx_c)
        assert np.allclose(cc.cov_tau, self.Mxy_c)

    def test_XY_weighted_meanconst(self):
        cc = covariance_lagged(data=self.data, remove_constant_mean=self.mean_const, c0t=True, weights=self.wobj, lag=self.lag,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.mx_c_wobj)
        assert np.allclose(cc.mean_tau, self.my_c_wobj)
        assert np.allclose(cc.cov, self.Mxx_c_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy_c_wobj)

    def test_XY_sym_meanconst(self):
        cc = covariance_lagged(data=self.data, remove_constant_mean=self.mean_const, c0t=True, reversible=True, lag=self.lag,
                               bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_c_sym)
        assert np.allclose(cc.cov, self.Mxx_c_sym)
        assert np.allclose(cc.cov_tau, self.Mxy_c_sym)

    def test_XY_sym_weighted_meanconst(self):
        cc = covariance_lagged(data=self.data, remove_constant_mean=self.mean_const, c0t=True, reversible=True, weights=self.wobj,
                               lag=self.lag, bessel=False, chunksize=self.chunksize)
        assert np.allclose(cc.mean, self.m_c_sym_wobj)
        assert np.allclose(cc.cov, self.Mxx_c_sym_wobj)
        assert np.allclose(cc.cov_tau, self.Mxy_c_sym_wobj)

if __name__ == "__main__":
    unittest.main()