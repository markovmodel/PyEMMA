
import unittest
from pyemma._base.estimator import Estimator
from pyemma.util.annotators import estimation_required, alias, aliased, deprecated


@aliased
class TestEstimator(Estimator):

    @estimation_required
    @alias('testimator_method_requires')
    def test_method_requires_estimation(self):
        pass

    @alias('testimator_method_requires_rev')
    @estimation_required
    def test_method_requires_estimation_reverse(self):
        pass

    def test_method_does_not_require_estimation(self):
        pass

    def _estimate(self, X):
        pass


class TestEstimationRequired(unittest.TestCase):

    def test_requires_estimation(self):
        testimator = TestEstimator()
        self.assertRaises(ValueError, testimator.test_method_requires_estimation)
        testimator.estimate(None)
        # now that we called 'estimate()', should not raise
        testimator.test_method_requires_estimation()

    def test_requires_estimation_alias(self):
        testimator = TestEstimator()
        self.assertRaises(ValueError, testimator.testimator_method_requires)
        testimator.estimate(None)
        # now that we called 'estimate()', should not raise
        testimator.testimator_method_requires()

    def test_requires_estimation_alias_reverse(self):
        testimator = TestEstimator()
        self.assertRaises(ValueError, testimator.testimator_method_requires_rev)
        testimator.estimate(None)
        # now that we called 'estimate()', should not raise
        testimator.testimator_method_requires_rev()

    def test_does_not_require_estimation(self):
        testimator = TestEstimator()
        # does not require 'estimate()', should not raise
        testimator.test_method_does_not_require_estimation()