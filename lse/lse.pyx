#
#   Copyright 2015 Christoph Wehmeyer
#

import numpy as np
cimport numpy as np

cdef extern from "_lse.h":
    double rc_logsumexp(double *array, int length)
    double rc_logsumexp_pair(double a, double b)

def logsumexp(np.ndarray[double, ndim=1, mode="c"] array not None):
    return rc_logsumexp(<double*> np.PyArray_DATA(array), array.shape[0])

def logsumexp_pair(a, b):
    return rc_logsumexp_pair(a, b)
