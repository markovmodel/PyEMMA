import unittest

from pyemma._base.estimator import Estimator

try:
    import sklearn

    have_sklearn = True
except ImportError:
    have_sklearn = False


class TestBaseEstimator(unittest.TestCase):

    @unittest.skipIf(not have_sklearn, 'no sklearn')
    def test_sklearn_compat_fit(self):
        class T(Estimator):
            def _estimate(self, X):
                return self

        from sklearn.pipeline import Pipeline
        p = Pipeline([('test', T())])
        p.fit([1, 2, 3])
