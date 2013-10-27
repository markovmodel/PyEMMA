'''
This module is used to initialize a global Java VM instance, to run the python
wrapper for the stallone library
Created on 15.10.2013

@author: marscher
'''
from log import log as _log
import numpy as _np

"""is the stallone python binding available?"""
stallone_available = None

try:
    _log.debug('try to initialize stallone module')
    from pystallone import *
    from pystallone.ArrayWrapper import ArrayWrapper as ArrayWrapper
    jenv = initVM(initialheap='32m', maxheap='512m')
    stallone_available = True
    _log.info('stallone initialized successfully.')
except ImportError:
    _log.error('stallone could not be found.')
    stallone_available = False
except ValueError as ve:
    stallone_available = False
    _log.error('java vm initialization for stallone went wrong: ', ve)
except:
    _log.error('unknown exception occured.')


def ndarray_to_stallone_array(ndarray):
    if not stallone_available:
        raise RuntimeError('stallone not available')
    
    shape = ndarray.shape
    dtype = ndarray.dtype
    factory = None
    cast_func = None
    
    if dtype == _np.float32 or dtype == _np.float64:
        factory = API.doublesNew
        cast_func = float
    elif dtype == _np.int32 or dtype == _np.int64:
        factory = API.intsNew
        cast_func = int
    else:
        raise TypeError('unsupported datatype: ', dtype)
    
    if len(shape) == 1:
        n = shape[0]
        v = factory.array(n)
        for i in xrange(n):
            v.set(i, cast_func(ndarray[i]))
        return v
    elif len(shape) == 2:
        n = shape[0]
        m = shape[1]
        try:
            A = factory.matrix(n, m)
        except AttributeError:
            A = factory.table(n, m)
        
        for i in xrange(n):
            for j in xrange(m):
                val = ndarray[i, j]
                A.set(i, j, cast_func(val))
        return A
    else:
        raise ValueError('unsupported shape: ', shape)