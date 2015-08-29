################################################################################
#
#   lse.pyx - logsumexp implementation in C (cython wrapper)
#
#   author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>
#
################################################################################

import numpy as np
cimport numpy as np

cdef extern from "_lse.h":
    void _sort(double *array, int L, int R)
    double _logsumexp(double *array, int length)
    double _logsumexp_pair(double a, double b)

# _sort()is based on examples from http://www.linux-related.de (2004)
def sort(np.ndarray[double, ndim=1, mode="c"] array not None):
    _sort(<double*> np.PyArray_DATA(array), 0, array.shape[0])

def logsumexp(np.ndarray[double, ndim=1, mode="c"] array not None):
    return _logsumexp(<double*> np.PyArray_DATA(array), array.shape[0])

def logsumexp_pair(a, b):
    return _logsumexp_pair(a, b)
