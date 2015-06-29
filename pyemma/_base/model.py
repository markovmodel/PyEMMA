__author__ = 'noe'

import numpy as _np

from pyemma._ext.sklearn.base import Parametric as _Parametric
from pyemma.util.statistics import confidence_interval
from pyemma.util.reflection import call_member

class Model(_Parametric):
    """ Base class for pyEMMA models

    """
    pass


class SampledModel(Model):

    def __init__(self, samples, conf=0.95):
        self.samples = samples
        self.conf = conf
        self._nsamples = len(samples)

    def set_confidence(self, conf):
        self.conf = conf

    @property
    def nsamples(self):
        """Number of model samples"""
        return self._nsamples

#    def mean_model(self):
#        """Computes the mean model from the given samples"""
#        raise NotImplementedError('mean_model is not implemented in class '+str(self.__class__))

    def sample_f(self, f, *args):
        return [call_member(M, f, *args) for M in self.samples]

    def sample_mean(self, f, *args):
        vals = self.sample_f(f, *args)
        return _np.mean(vals, axis=0)

    def sample_std(self, f, *args):
        vals = self.sample_f(f, *args)
        return _np.std(vals, axis=0)

    def sample_conf(self, f, *args):
        vals = self.sample_f(f, *args)
        return confidence_interval(vals, conf=self.conf)
