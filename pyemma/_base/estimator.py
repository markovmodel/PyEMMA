__author__ = 'noe'

class Estimator:
    """ Base class for pyEMMA estimators

    """

    def estimate(self, **kwargs):
        raise NotImplementedError('The estimate() method is not implemented for this Estimator.')
