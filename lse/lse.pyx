#
#   Copyright 2015 Christoph Wehmeyer
#

import numpy as np
cimport numpy as np

cdef extern from "_lse.h":
    double _logsumexp(double *array, int length)
    double _logsumexp_pair(double a, double b)

def logsumexp(np.ndarray[double, ndim=1, mode="c"] array not None):
    return _logsumexp(<double*> np.PyArray_DATA(array), array.shape[0])

def logsumexp_pair(a, b):
    return _logsumexp_pair(a, b)
