
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



import inspect
import sys
import os

from pyemma._ext.sklearn.base import BaseEstimator as _BaseEstimator
from pyemma._ext.sklearn.parameter_search import ParameterGrid
from pyemma.util import types as _types

# imports for external usage
from pyemma._ext.sklearn.base import clone as clone_estimator
from pyemma._base.loggable import Loggable

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
    >>> for p in grid: print(sorted(p)) # doctest: +SKIP
    {'reversible': False, 'lag': 1}
    {'reversible': True, 'lag': 1}
    {'reversible': False, 'lag': 10}
    {'reversible': True, 'lag': 10}
    {'reversible': False, 'lag': 100}
    {'reversible': True, 'lag': 100}

    """
    return ParameterGrid(pargrid)


def _call_member(obj, name, failfast=True, *args, **kwargs):
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
    args : list, optional, default=[]
        Arguments to be passed to the method (if any)

    kwargs: dict
    """
    try:
        attr = getattr(obj, name)
    except AttributeError as e:
        if failfast:
            raise e
        else:
            return None
    try:
        if inspect.ismethod(attr):  # call function
            return attr(*args, **kwargs)
        elif isinstance(attr, property):  # call property
                return obj.attr
        else:  # now it's an Attribute, so we can just return its value
            return attr
    except Exception as e:
        if failfast:
            raise e
        else:
            return None


def _estimate_param_scan_worker(estimator, params, X, evaluate, evaluate_args,
                                failfast, return_exceptions):
    """ Method that runs estimation for several parameter settings.

    Defined as a worker for parallelization

    """
    # run estimation
    model = None
    try:  # catch any exception
        estimator.estimate(X, **params)
        model = estimator.model
    except KeyboardInterrupt:
        # we want to be able to interactively interrupt the worker, no matter of failfast=False.
        raise
    except:
        e = sys.exc_info()[1]
        if isinstance(estimator, Loggable):
            estimator.logger.warning("Ignored error during estimation: %s" % e)
        if failfast:
            raise  # re-raise
        elif return_exceptions:
            model = e
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
        for ieval, name in enumerate(evaluate):
            # get method/attribute name and arguments to be evaluated
            #name = evaluate[ieval]
            args = ()
            if evaluate_args is not None:
                args = evaluate_args[ieval]
                # wrap single arguments in an iterable again to pass them.
                if _types.is_string(args):
                    args = (args, )
            # evaluate
            try:
                # try calling method/property/attribute
                value = _call_member(estimator.model, name, failfast, *args)
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
                        return_estimators=False, n_jobs=1, progress_reporter=None, show_progress=True,
                        return_exceptions=False):
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

    evaluate : str or list of str, optional
        The given methods or properties will be called on the estimated
        models, and their results will be returned instead of the full models.
        This may be useful for reducing memory overhead.

    evaluate_args: iterable of iterable, optional
        Arguments to be passed to evaluated methods. Note, that size has to match to the size of evaluate.

    failfast : bool
        If True, will raise an exception when estimation failed with an exception
        or trying to calls a method that doesn't exist. If False, will simply
        return None in these cases.

    return_estimators: bool
        If True, return a list estimators in addition to the models.

    show_progress: bool
        if the given estimator supports show_progress interface, we set the flag
        prior doing estimations.

    return_exceptions: bool, default=False
        if failfast is False while this setting is True, returns the exception thrown at the actual grid element,
        instead of None.

    Returns
    -------
    models : list of model objects or evaluated function values
        A list of estimated models in the same order as param_sets. If evaluate
        is given, each element will contain the results from these method
        evaluations.

    estimators (optional) : list of estimator objects. These are returned only
        if return_estimators=True

    Examples
    --------

    Estimate a maximum likelihood Markov model at lag times 1, 2, 3.

    >>> from pyemma.msm.estimators import MaximumLikelihoodMSM, BayesianMSM
    >>>
    >>> dtraj = [0,0,1,2,1,0,1,0,1,2,2,0,0,0,1,1,2,1,0,0,1,2,1,0,0,0,1,1,0,1,2]  # mini-trajectory
    >>> param_sets=param_grid({'lag': [1,2,3]})
    >>>
    >>> estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets, evaluate='timescales')
    [array([ 1.24113168,  0.77454377]), array([ 2.65266698,  1.42909842]), array([ 5.34810405,  1.14784446])]

    Now we also want to get samples of the timescales using the BayesianMSM.
    >>> estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets, failfast=False,
    ...     evaluate=['timescales', 'timescales_samples']) # doctest: +SKIP
    [[array([ 1.24113168,  0.77454377]), None], [array([ 2.48226337,  1.54908754]), None], [array([ 3.72339505,  2.32363131]), None]]

    We get Nones because the MaximumLikelihoodMSM estimator doesn't provide timescales_samples. Use for example
    a Bayesian estimator for that.

    Now we also want to get samples of the timescales using the BayesianMSM.
    >>> estimate_param_scan(BayesianMSM, dtraj, param_sets, show_progress=False,
    ...     evaluate=['timescales', 'sample_f'], evaluate_args=((), ('timescales', ))) # doctest: +SKIP
    [[array([ 1.24357685,  0.77609028]), [array([ 1.5963252 ,  0.73877883]), array([ 1.29915847,  0.49004912]), array([ 0.90058583,  0.73841786]), ... ]]

    """
    # make sure we have an estimator object
    estimator = get_estimator(estimator)
    if hasattr(estimator, 'show_progress'):
        estimator.show_progress = show_progress

    if n_jobs is None:
        from pyemma._base.parallel import get_n_jobs
        n_jobs = get_n_jobs(logger=getattr(estimator, 'logger', None))

    # if we want to return estimators, make clones. Otherwise just copy references.
    # For parallel processing we always need clones.
    # Also if the Estimator is its own Model, we have to clone.
    from pyemma._base.model import Model
    if (return_estimators or
        n_jobs > 1 or n_jobs is None or
        isinstance(estimator, Model)):
        estimators = [clone_estimator(estimator) for _ in param_sets]
    else:
        estimators = [estimator for _ in param_sets]

    # only show progress of parameter study.
    if hasattr(estimators[0], 'show_progress'):
        for e in estimators:
            e.show_progress = False

    # if we evaluate, make sure we have a list of functions to evaluate
    if _types.is_string(evaluate):
        evaluate = [evaluate]
    if _types.is_string(evaluate_args):
        evaluate_args = [evaluate_args]

    if evaluate is not None and evaluate_args is not None and len(evaluate) != len(evaluate_args):
        raise ValueError("length mismatch: evaluate ({}) and evaluate_args ({})".format(len(evaluate), len(evaluate_args)))

    logger_available = hasattr(estimators[0], 'logger')
    if logger_available:
        logger = estimators[0].logger
    if progress_reporter is None:
        from unittest.mock import MagicMock
        ctx = progress_reporter = MagicMock()
        callback = None
    else:
        ctx = progress_reporter._progress_context('param-scan')
        callback = lambda _: progress_reporter._progress_update(1, stage='param-scan')

        progress_reporter._progress_register(len(estimators), stage='param-scan',
                                             description="estimating %s" % str(estimator.__class__.__name__))

    # TODO: test on win, osx
    if n_jobs > 1 and os.name == 'posix':
        if logger_available:
            logger.debug('estimating %s with n_jobs=%s', estimator, n_jobs)
        # iterate over parameter settings
        task_iter = ((estimator,
                      param_set, X,
                      evaluate,
                      evaluate_args,
                      failfast, return_exceptions)
                     for estimator, param_set in zip(estimators, param_sets))

        from pathos.multiprocessing import Pool
        pool = Pool(processes=n_jobs)
        args = list(task_iter)

        from contextlib import closing

        def error_callback(*args, **kw):
            if failfast:
                # TODO: can we be specific here? eg. obtain the stack of the actual process or is this the master proc?
                raise Exception('something failed')

        with closing(pool), ctx:
            res_async = [pool.apply_async(_estimate_param_scan_worker, a, callback=callback,
                                          error_callback=error_callback) for a in args]
            res = [x.get() for x in res_async]

    # if n_jobs=1 don't invoke the pool, but directly dispatch the iterator
    else:
        if logger_available:
            logger.debug('estimating %s with n_jobs=1 because of the setting or '
                         'you not have a POSIX system', estimator)
        res = []
        with ctx:
            for estimator, param_set in zip(estimators, param_sets):
                res.append(_estimate_param_scan_worker(estimator, param_set, X,
                                                       evaluate, evaluate_args, failfast, return_exceptions))
                if progress_reporter is not None:
                    progress_reporter._progress_update(1, stage='param-scan')

    # done
    if return_estimators:
        return res, estimators
    else:
        return res


# we do not want to derive from Serializable here, because this would make all children serializable.
# However we guide serializable children, what to store/restore.
class Estimator(_BaseEstimator, Loggable):
    """ Base class for PyEMMA estimators

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
        # ensure _estimate returned something
        assert self._model is not None
        self._estimated = True
        return self

    def _estimate(self, X):
        raise NotImplementedError(
            'You need to overload the _estimate() method in your Estimator implementation!')

    def fit(self, X, y=None):
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
        return self.estimate(X)

    @property
    def model(self):
        """The model estimated by this Estimator"""
        try:
            if self._model is None:
                raise AttributeError
            return self._model
        except AttributeError:
            raise AttributeError(
                'Model has not yet been estimated. Call estimate(X) or fit(X) first')

    def _check_estimated(self):
        if not self._estimated:
            raise Exception('Estimator is not estimated. Call estimate/fit or partial_fit first.')

    # override get_params here, to handle PyEMMA_DeprecationWarnings as well
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        import warnings
        from pyemma.util.exceptions import PyEMMA_DeprecationWarning

        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            warnings.simplefilter("always", PyEMMA_DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category in (DeprecationWarning, PyEMMA_DeprecationWarning):
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = list(value.get_params().items())
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    # serialization handling
    __serialize_fields = ('_estimated', 'model')

    def __my_getstate__(self):
        state = {}

        inspect_classes = filter(lambda c: hasattr(c, '_get_param_names'), self.__class__.__mro__)
        for c in inspect_classes:
            state.update({k: getattr(self, k, None) for k in c._get_param_names()})

        return state

    def __my_setstate__(self, state):
        if state:
            valid_parameters = list()
            for c in filter(lambda c: hasattr(c, '_get_param_names'), self.__class__.__mro__):
                valid_parameters.extend(c._get_param_names())
            for param in valid_parameters:
                if param in state:
                    setattr(self, param, state.get(param))
