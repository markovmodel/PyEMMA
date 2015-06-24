__author__ = 'noe'

from pyemma._ext.sklearn.base import BaseEstimator as _SklearnEstimator
from pyemma._ext.sklearn.parameter_search import ParameterGrid
from pyemma.util.log import getLogger
import inspect
from pyemma.util import types as _types

def get_estimator(estimator):
    """ Returns an estimator object given an estimator object or class

    Parameters
    ----------
    estimator : Estimator class or object

    Returns
    -------
    estimator : Estimator object

    """
    if inspect.isclass(estimator):
        estimator = estimator()  # construct the estimator with default settings
    return estimator

def param_grid(pargrid):
    """ Generates an iterable over all possible parameter combinations from the grid

    Parameters
    ----------
    pargrid : dictionary with lists where multiple values are wanted

    Examples
    --------
    Generates parameter sets with different lag times:

    >>> grid = param_grid({'lag':[1,2,5,10,20,50]})
    >>> for p in grid: print p
    {'lag': 1}
    {'lag': 2}
    {'lag': 5}
    {'lag': 10}
    {'lag': 20}
    {'lag': 50}

    Generates parameter sets with all combinations of several parameter values:

    >>> grid = param_grid({'lag':[1,10,100], 'reversible':[False,True]})
    >>> for p in grid: print p
    {'reversible': False, 'lag': 1}
    {'reversible': True, 'lag': 1}
    {'reversible': False, 'lag': 10}
    {'reversible': True, 'lag': 10}
    {'reversible': False, 'lag': 100}
    {'reversible': True, 'lag': 100}

    """
    return ParameterGrid(pargrid)

def _call_member(obj, name, failfast=True):
    """ Calls the specified method, property or attribute of the given object

    Parameters
    ----------
    obj : object
        The object that will be used
    name : str
        Name of method, property or attribute
    failfast : bool
        If True, will raise an exception when trying a method that doesn't exist. If False, will simply return None
        in that case
    """
    try:
        method = getattr(obj, name)
    except AttributeError as e:
        if failfast:
            raise e
        else:
            return None
    if str(type(method)) == '<type \'instancemethod\'>':  # call function without params
        return obj.method()
    elif str(type(method)) == '<type \'property\'>':  # call property
        return obj.method
    else:  # now it's an Attribute, so we can just return its value
        return method

def estimate_param_scan(estimator, X, param_sets, evaluate=None, failfast=True):
    # TODO: parallelize. For options see http://scikit-learn.org/stable/modules/grid_search.html
    # TODO: allow to specify method parameters in evaluate
    """ Runs multiple estimations using a list of parameter settings

    Parameters
    ----------
    estimator : Estimator object or class
        An estimator object that provides an estimate(X, **params) function. If only a class is provided here,
        the Estimator objects will be constructed with default parameter settings, and the parameter settings
        from param_sets for each estimation. If you want to specify other parameter settings for those parameters
        not specified in param_sets, construct an Estimator before and pass the object.
    param_sets : iterable over dictionaries
        An iterable that provides parameter settings. Each element defines a parameter set, for which an estimation
        will be run using these parameters in estimate(X, **params). All other parameter settings will be taken from
        the default settings in the estimator object.
    evaluate : str or list of str
        The given methods or properties will be called on the estimated models, and their results will be returned
        instead of the full models. This may be useful for reducing memory overhead.
    failfast : bool
        If True, will raise an exception when trying a method that doesn't exist. If False, will simply return None
        in that case

    Return
    ------
    models : list of model objects or evaluated function values
        A list of estimated models in the same order as param_sets. If evaluate is given, each element will contain
        the results from these method evaluations.

    Examples
    --------

    Estimate a maximum likelihood Markov model at lag times 1, 2, 3.

    >>> from pyemma.msm.estimators import MaximumLikelihoodMSM
    >>>
    >>> dtraj = [0,0,1,2,1,0,1,0,1,2,2,0,0,0,1,1,2,1,0,0,1,2,1,0,0,0,1,1,0,1,2]  # mini-trajectory
    >>> param_sets=param_grid({'lag': [1,2,3]})
    >>>
    >>> estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets, evaluate='timescales')
    [array([ 1.24113167,  0.77454377]), array([ 2.65266703,  1.42909841]), array([ 5.34810395,  1.14784446])]

    Try also getting samples of the timescales

    >>> estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets, evaluate=['timescales', 'timescales_samples'])
    [[array([ 1.24113167,  0.77454377]), None], [array([ 2.65266703,  1.42909841]), None], [array([ 5.34810395,  1.14784446]), None],

    We get Nones because the MaximumLikelihoodMSM estimator doesn't provide timescales_samples. Use for example
    a Bayesian estimator for that.

    """
    # TODO: parallelization should first clone estimators and dispatch estimation routines.
    # make sure we have an estimator object
    estimator = get_estimator(estimator)
    res = []  # prepare results

    # if we evaluate, make sure we have a list of functions to evaluate
    if _types.is_string(evaluate):
        evaluate = [evaluate]

    # iterate over parameter settings
    for p in param_sets:
        # run estimation
        model = estimator.estimate(X, **p)

        # deal with result
        if evaluate is None:  # we want full models
            res.append(model)
        elif _types.is_iterable(evaluate):  # we want to evaluate function(s) for each model
            values = []  # the function values fo each model
            for name in evaluate:
                try:
                    value = _call_member(model, name)  # try calling method/property/attribute
                except AttributeError as e:  # couldn't find method/property/attribute
                    if failfast:
                        raise e  # raise an AttributeError
                    else:
                        value = None  # we just ignore it and return None
                values.append(value)
            # if we only have one value, unpack it
            if len(values) == 1:
                values = values[0]
            res.append(values)
        else:
            raise ValueError('Invalid setting for evaluate: '+str(evaluate))

    # done
    return res


class Estimator(_SklearnEstimator):
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
        **params : New estimation parameter values. The parameters must that have been announced in the
            __init__ method of this estimator. The present settings will overwrite the settings of parameters
            given in the __init__ method, i.e. the parameter values after this call will be those that have been
            used for this estimation. Use this option if only one or a few parameters change with respect to
            the __init__ settings for this run, and if you don't need to remember the original settings of these
            changed parameters.

        Returns
        -------
        model : object
            The estimated model.

        """
        # set params
        if params:
            self.set_params(**params)
        self._model = self._estimate(X)
        return self._model

    def _estimate(self, X):
        raise NotImplementedError('You need to overload the _estimate() method in your Estimator implementation!')

    def fit(self, X):
        """ For compatibility with sklearn.
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
