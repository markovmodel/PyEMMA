__author__ = 'noe'

from pyemma._ext.sklearn.base import BaseEstimator as SklearnEstimator

class Estimator(SklearnEstimator):
    """ Base class for pyEMMA estimators

    """

    def estimate(self, **kwargs):
        raise NotImplementedError('The estimate() method is not implemented for this Estimator.')
