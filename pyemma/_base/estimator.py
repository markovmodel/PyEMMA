
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, print_function

from six.moves import range
import inspect, sys

from pyemma._ext.sklearn.base import BaseEstimator as _BaseEstimator
from pyemma._ext.sklearn.parameter_search import ParameterGrid
from pyemma.util import types as _types

# imports for external usage
from pyemma._ext.sklearn.base import clone as clone_estimator
from pyemma._base.logging import Loggable

__author__ = 'noe, marscher'


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
        # construct the estimator with default settings
        estimator = estimator()
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
    >>> for p in grid: print(p)
    {'lag': 1}
    {'lag': 2}
    {'lag': 5}
    {'lag': 10}
    {'lag': 20}
    {'lag': 50}

    Generates parameter sets with all combinations of several parameter values:

    >>> grid = param_grid({'lag':[1,10,100], 'reversible':[False,True]})
    >>> for p in grid: print(p)
    {'reversible': False, 'lag': 1}
    {'reversible': True, 'lag': 1}
    {'reversible': False, 'lag': 10}
    {'reversible': True, 'lag': 10}
    {'reversible': False, 'lag': 100}
    {'reversible': True, 'lag': 100}

    """
    return ParameterGrid(pargrid)


def _call_member(obj, name, args=None, failfast=True):
    """ Calls the specified method, property or attribute of the given object

    Parameters
    ----------
    obj : object
        The object that will be used
    name : str
        Name of method, property or attribute
    args : dict, optional, default=None
        Arguments to be passed to the method (if any)
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

    if inspect.ismethod(object):  # call function
        if args is None:
            return method()
        else:
            return method(*args)
    elif isinstance(type(obj).name, property):  # call property
        return method
    else:  # now it's an Attribute, so we can just return its value
        return method


def _estimate_param_scan_worker(estimator, params, X, evaluate, evaluate_args,
                                failfast):
    """ Method that runs estimation for several parameter settings.

    Defined as a worker for Parallelization

    """
    # run estimation
    model = None
    try:  # catch any exception
        estimator.estimate(X, **params)
        model = estimator.model
    except:
        e = sys.exc_info()[1]
        if isinstance(estimator, Loggable):
            estimator.logger.warning("Ignored error during estimation: %s" % e)
        if failfast:
            raise  # re-raise
        else:
            pass  # just return model=None

    # deal with results
    res = []

    # deal with result
    if evaluate is None:  # we want full models
        res.append(model)
    # we want to evaluate function(s) of the model
    elif _types.is_iterable(evaluate):
        values = []  # the function values the model
        for ieval in range(len(evaluate)):
            # get method/attribute name and arguments to be evaluated
            name = evaluate[ieval]
            args = None
            if evaluate_args is not None:
                args = evaluate_args[ieval]
            # evaluate
            try:
                # try calling method/property/attribute
                value = _call_member(estimator.model, name, args=args)
            # couldn't find method/property/attribute
            except AttributeError as e:
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
        raise ValueError('Invalid setting for evaluate: ' + str(evaluate))

    if len(res) == 1:
        res = res[0]
    return res


def estimate_param_scan(estimator, X, param_sets, evaluate=None, evaluate_args=None, failfast=True,
                        return_estimators=False, n_jobs=1, progress_reporter=None):
    """ Runs multiple estimations using a list of parameter settings

    Parameters
    ----------
    estimator : Estimator object or class
        An estimator object that provides an estimate(X, **params) function.
        If only a class is provided here, the Estimator objects will be
        constructed with default parameter settings, and the parameter settings
        from param_sets for each estimation. If you want to specify other
        parameter settings for those parameters not specified in param_sets,
        construct an Estimator before and pass the object.

    param_sets : iterable over dictionaries
        An iterable that provides parameter settings. Each element defines a
        parameter set, for which an estimation will be run using these
        parameters in estimate(X, **params). All other parameter settings will
        be taken from the default settings in the estimator object.

    evaluate : str or list of str
        The given methods or properties will be called on the estimated
        models, and their results will be returned instead of the full models.
        This may be useful for reducing memory overhead.

    failfast : bool
        If True, will raise an exception when estimation failed with an exception
        or trying to calls a method that doesn't exist. If False, will simply
        return None in these cases.

    Return
    ------
    models : list of model objects or evaluated function values
        A list of estimated models in the same order as param_sets. If evaluate
        is given, each element will contain the results from these method
        evaluations.

    estimators (optional) : list of estimator objects. These are returned only
        if return_estimators=True

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
    # make sure we have an estimator object
    estimator = get_estimator(estimator)
    # if we want to return estimators, make clones. Otherwise just copy references.
    # For parallel processing we always need clones
    if return_estimators or n_jobs > 1 or n_jobs is None:
        estimators = [clone_estimator(estimator) for _ in param_sets]
    else:
        estimators = [estimator for _ in param_sets]

    # if we evaluate, make sure we have a list of functions to evaluate
    if _types.is_string(evaluate):
        evaluate = [evaluate]

    # set call back for joblib
    if progress_reporter is not None:
        progress_reporter._progress_register(len(estimators), stage=0,
                                             description="estimating %s" % str(estimator.__class__.__name__))

        if n_jobs > 1:
            class CallBack(object):
                def __init__(self, index, parallel):
                    self.index = index
                    self.parallel = parallel
                    self.reporter = progress_reporter

                def __call__(self, index):
                    if self.reporter is not None:
                        self.reporter._progress_update(1, stage=0)
                    if self.parallel._original_iterable:
                        self.parallel.dispatch_next()
            import joblib.parallel
            joblib.parallel.CallBack = CallBack
        else:
            def _print(msg, msg_args):
                # NOTE: this is a ugly hack, because if we only use one job,
                # we do not get the joblib callback interface, as a workaround
                # we use the Parallel._print function, which is called with
                # msg_args = (done_jobs, total_jobs)
                if len(msg_args) == 2:
                    progress_reporter._progress_update(1, stage=0)

    # iterate over parameter settings
    from joblib import Parallel
    import joblib
    pool = Parallel(n_jobs=n_jobs)

    if progress_reporter is not None and n_jobs == 1:
        pool._print = _print
        # NOTE: verbose has to be set, otherwise our print hack does not work.
        pool.verbose = 50

    task_iter = (joblib.delayed(_estimate_param_scan_worker)(estimators[i],
                                                             param_sets[i], X,
                                                             evaluate,
                                                             evaluate_args,
                                                             failfast,
                                                             )
                 for i in range(len(param_sets)))

    # container for model or function evaluations
    res = pool(task_iter)

    if progress_reporter is not None:
        progress_reporter._progress_force_finish(0)

    # done
    if return_estimators:
        return res, estimators
    else:
        return res


class Estimator(_BaseEstimator, Loggable):
    """ Base class for pyEMMA estimators

    """
    # flag indicating if estimator's estimate method has been called
    _estimated = False

    def estimate(self, X, **params):
        """ Estimates the model given the data X

        Parameters
        ----------
        X : object
            A reference to the data from which the model will be estimated
        params : dict
            New estimation parameter values. The parameters must that have been
            announced in the __init__ method of this estimator. The present
            settings will overwrite the settings of parameters given in the
            __init__ method, i.e. the parameter values after this call will be
            those that have been used for this estimation. Use this option if
            only one or a few parameters change with respect to
            the __init__ settings for this run, and if you don't need to
            remember the original settings of these changed parameters.

        Returns
        -------
        estimator : object
            The estimated estimator with the model being available.

        """
        # set params
        if params:
            self.set_params(**params)
        self._model = self._estimate(X)
        self._estimated = True
        return self

    def _estimate(self, X):
        raise NotImplementedError(
            'You need to overload the _estimate() method in your Estimator implementation!')

    def fit(self, X):
        """Estimates parameters - for compatibility with sklearn.

        Parameters
        ----------
        X : object
            A reference to the data from which the model will be estimated

        Returns
        -------
        estimator : object
            The estimator (self) with estimated model.

        """
        self.estimate(X)
        return self

    @property
    def model(self):
        """The model estimated by this Estimator"""
        try:
            return self._model
        except AttributeError:
            raise AttributeError(
                'Model has not yet been estimated. Call estimate(X) or fit(X) first')
