__author__ = 'noe'

from pyemma._ext.sklearn.base import BaseEstimator as SklearnEstimator
from pyemma.util.log import getLogger

class Estimator(SklearnEstimator):
    """ Base class for pyEMMA estimators

    """

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    @property
    def logger(self):
        """ The logger for this Estimator """
        try:
            return self._logger
        except AttributeError:
            self.__create_logger()
            return self._logger

    def estimate(self, X, **params):
        """ Estimates the model given the data X

        Parameters
        ----------
        X : object
            A reference to the data from which the model will be estimated
        **params : Estimation parameters for this estimation. Must be parameters that have been announced in the
            __init__ method of this estimator. The present settings will overwrite the settings of parameters
            given in the __init__ method and will be used for this estimation run. Use this option if only
            one or a few parameters change with respect to the __init__ settings for this run.

        Returns
        -------
        model : object
            The estimated model.

        """
        # set params
        if params:
            self.set_params(params)
        self._model = self._estimate(X)

    def _estimate(self, X):
        raise NotImplementedError('The _estimate() method is not implemented for this Estimator.')

    def fit(self, X):
        """ For compatibility with sklearn. Requires that X has been announced in the constructor
        :param X:
        :return:
        """
        self.estimate(X)

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            raise AttributeError('Model has not yet been estimated. Call estimate(X) or fit(X) first')
