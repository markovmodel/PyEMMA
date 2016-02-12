
import unittest
from pyemma._base.estimator import Estimator
from pyemma.util.annotators import estimation_required, alias, aliased, deprecated


@aliased
class TestEstimator(Estimator):

    def __init__(self):
        self._prop = ""

    @property
    @estimation_required
    def property_method_requires(self):
        return self._prop

    @deprecated("testimator_method_requires is deprecated.")
    @estimation_required
    @alias('testimator_method_requires')
    def method_requires_estimation(self):
        pass

    @alias('testimator_method_requires_rev')
    @estimation_required
    @deprecated("method_requires_estimation_reverse is deprecated.")
    def method_requires_estimation_reverse(self):
        pass

    def method_does_not_require_estimation(self):
        pass

    def _estimate(self, X):
        return self

@deprecated
def _deprecated_method():
    pass

class TestEstimationRequired(unittest.TestCase):

    def test_deprecated_method(self):
        # sanity test
        _deprecated_method()

    def test_requires_estimation_property(self):
        testimator = TestEstimator()
        with self.assertRaises(ValueError) as ctx:
            testimator.property_method_requires
        self.assertTrue('Tried calling property_method_requires on TestEstimator' in str(ctx.exception))
        testimator.estimate(None)
        self.assertEqual("", testimator.property_method_requires, "should return an empty string since now estimated")

    def test_requires_estimation(self):
        testimator = TestEstimator()
        self.assertRaises(ValueError, testimator.method_requires_estimation)
        testimator.estimate(None)
        # now that we called 'estimate()', should not raise
        testimator.method_requires_estimation()

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
        testimator.method_does_not_require_estimation()

