__author__ = 'noe'

from pyemma._ext.sklearn.base import BaseEstimator as SklearnEstimator

class Estimator(SklearnEstimator):
    """ Base class for pyEMMA estimators

    """

    def set_data(self, X):
        """ Sets the data on which the estimation will be run.

        Do any necessary updated upon setting or changing the data here.

        Parameters
        ----------
        X : object
            A reference to the data set.

        """
        raise NotImplementedError('The set_data() method is not implemented for this Estimator.')

    def get_data(self, X):
        """ Gets a reference to the data on which the estimation is run.

        Do any necessary updated upon setting or changing the data here.

        Returns
        -------
        X : object
            A reference to the data set.

        """
        raise NotImplementedError('The get_data() method is not implemented for this Estimator.')

    def estimate(self, **kwargs):
        raise NotImplementedError('The estimate() method is not implemented for this Estimator.')

    def fit(self, X):
        """ For compatibility with sklearn. Requires that X has been announced in the constructor
        :param X:
        :return:
        """
        self.set_data(X)
        self.estimate()