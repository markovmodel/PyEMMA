
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

import numpy as _np
import warnings

from pyemma._ext.sklearn.base import _pprint
from pyemma.util.statistics import confidence_interval
from pyemma.util.reflection import call_member, getargspec_no_self

__author__ = 'noe'


class Model(object):
    """ Base class for PyEMMA models

    This class is inspired by sklearn's BaseEstimator class. However, we define parameter names not by the
    current class' __init__ but have to announce them. This allows us to also remember the parameters of model
    superclasses. This class can be mixed with pyEMMA and sklearn Estimators.

    """

    def __my_getstate__(self):
        state = {}

        inspect_classes = filter(lambda c: hasattr(c, '_get_model_param_names'), self.__class__.__mro__)
        for c in inspect_classes:
            state.update({k: getattr(self, k, None) for k in c._get_model_param_names()})

        return state

    def __my_setstate__(self, state):
        if state:
            for c in filter(lambda c: hasattr(c, '_get_model_param_names'), self.__class__.__mro__):
                # TODO: actually we would desire to pop from state, but this can't be done because of ThermoMSM (would pop pi twice)
                params_for_c = {k: state.get(k) for k in c._get_model_param_names()}
                c.set_model_params(self, **params_for_c)

    @classmethod
    def _get_model_param_names(cls):
        r"""Get parameter names for the model"""
        # fetch model parameters
        if hasattr(cls, 'set_model_params'):
            # introspect the constructor arguments to find the model parameters
            # to represent
            args, varargs, kw, default = getargspec_no_self(cls.set_model_params)
            if varargs is not None:
                raise RuntimeError("PyEMMA models should always specify their parameters in the signature"
                                   " of their set_model_params (no varargs). %s doesn't follow this convention."
                                   % (cls,))
            return args
        else:
            # No parameters known
            return []

    def set_model_params(self, **kw):
        for k in kw:
            setattr(self, k, kw[k])

    def update_model_params(self, **params):
        r"""Update given model parameter if they are set to specific values"""
        for key, value in params.items():
            if not hasattr(self, key):
                setattr(self, key, value)  # set parameter for the first time.
            elif getattr(self, key) is None:
                setattr(self, key, value)  # update because this parameter is still None.
            elif value is not None:
                setattr(self, key, value)  # only overwrite if set to a specific value (None does not overwrite).

    def get_model_params(self, deep=True):
        r"""Get parameters for this model.

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
        for key in self._get_model_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            from pyemma.util.exceptions import PyEMMA_DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            warnings.simplefilter("always", PyEMMA_DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category in(DeprecationWarning, PyEMMA_DeprecationWarning):
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

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_model_params(deep=False),
                                               offset=len(class_name),),)


class SampledModel(Model):

    def __init__(self, samples, conf=0.95):
        self.set_model_params(samples=samples, conf=conf)

    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, samples=None, conf=0.95):
        self.update_model_params(samples=samples, conf=conf)

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        if value is not None:
            self.nsamples = len(value)
        self._samples = value

    def _check_samples_available(self):
        if self.samples is None:
            raise AttributeError('Model samples not available in '+str(self)+'. Call set_model_params with samples.')

    def sample_f(self, f, *args, **kwargs):
        r"""Evaluated method f for all samples

        Calls f(\*args, \*\*kwargs) on all samples.

        Parameters
        ----------
        f : method reference or name (str)
            Model method to be evaluated for each model sample

        args : arguments
            Non-keyword arguments to be passed to the method in each call

        kwargs : keyword-argments
            Keyword arguments to be passed to the method in each call

        Returns
        -------
        vals : list
            list of results of the method calls

        """
        self._check_samples_available()
        # TODO: can we use np.fromiter here? We would ne the same shape of every member for this!
        return [call_member(M, f, *args, **kwargs) for M in self.samples]

    def sample_mean(self, f, *args, **kwargs):
        r"""Sample mean of numerical method f over all samples

        Calls f(\*args, \*\*kwargs) on all samples and computes the mean.
        f must return a numerical value or an ndarray.

        Parameters
        ----------
        f : method reference or name (str)
            Model method to be evaluated for each model sample
        args : arguments
            Non-keyword arguments to be passed to the method in each call
        kwargs : keyword-argments
            Keyword arguments to be passed to the method in each call

        Returns
        -------
        mean : float or ndarray
            mean value or mean array

        """
        vals = self.sample_f(f, *args, **kwargs)
        return _np.mean(vals, axis=0)

    def sample_std(self, f, *args, **kwargs):
        r"""Sample standard deviation of numerical method f over all samples

        Calls f(\*args, \*\*kwargs) on all samples and computes the standard deviation.
        f must return a numerical value or an ndarray.

        Parameters
        ----------
        f : method reference or name (str)
            Model method to be evaluated for each model sample
        args : arguments
            Non-keyword arguments to be passed to the method in each call
        kwargs : keyword-argments
            Keyword arguments to be passed to the method in each call

        Returns
        -------
        std : float or ndarray
            standard deviation or array of standard deviations

        """
        vals = self.sample_f(f, *args, **kwargs)
        return _np.std(vals, axis=0)

    def sample_conf(self, f, *args, **kwargs):
        r"""Sample confidence interval of numerical method f over all samples

        Calls f(\*args, \*\*kwargs) on all samples and computes the confidence interval.
        Size of confidence interval is given in the construction of the
        SampledModel. f must return a numerical value or an ndarray.

        Parameters
        ----------
        f : method reference or name (str)
            Model method to be evaluated for each model sample
        args : arguments
            Non-keyword arguments to be passed to the method in each call
        kwargs : keyword-argments
            Keyword arguments to be passed to the method in each call

        Returns
        -------
        L : float or ndarray
            lower value or array of confidence interval
        R : float or ndarray
            upper value or array of confidence interval

        """
        vals = self.sample_f(f, *args, **kwargs)
        return confidence_interval(vals, conf=self.conf)
