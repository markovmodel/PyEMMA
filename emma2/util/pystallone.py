'''
This module is used to initialize a global Java VM instance, to run the python
wrapper for the stallone library
Created on 15.10.2013

@author: marscher
'''
from log import getLogger
_log = getLogger()
import numpy as _np

"""is the stallone python binding available?"""
stallone_available = False

try:
    _log.debug('try to initialize stallone module')
    from stallone import *
    # todo: store and read jvm parameters in emma2.cfg
    jenv = initVM(initialheap='32m', maxheap='512m')
    stallone_available = True
    _log.info('stallone initialized successfully.')
except ImportError as ie:
    _log.error('stallone could not be found: %s' % ie)
except ValueError as ve:
    _log.error('java vm initialization for stallone went wrong: %s' % ve)
except BaseException as e:
    _log.error('unknown exception occurred: %s' %e)


def ndarray_to_stallone_array(ndarray):
    """
        Parameters
        ----------
        ndarray : 
        Returns
        -------
        IDoubleArray or IIntArray depending on input type
    """
    if not stallone_available:
        raise RuntimeError('stallone not available')
    
    _log.debug('called ndarray_to_stallone_array()')
    
    shape = ndarray.shape
    dtype = ndarray.dtype
    factory = None
    cast_func = None
    
    _log.debug("type of ndarray: %s" % dtype)
    
    if dtype == _np.float32 or dtype == _np.float64:
        factory = API.doublesNew
        cast_func = 'double'
    elif dtype == _np.int32 or dtype == _np.int64:
        factory = API.intsNew
        cast_func = 'int'
    else:
        raise TypeError('unsupported datatype: ', dtype)
    
    if len(shape) == 1:
        _log.debug('creating java vector.')
        # create a JArrayWrapper
        
        jarr = JArray(cast_func)(ndarray)
        v = factory.arrayFrom(jarr)
        _log.debug('finished java vector.')
        return v
    elif len(shape) == 2:
        _log.debug('converting to JArray matrix.')
        _log.debug('cast func: %s' %cast_func)
        _log.debug(type(ndarray))
        _log.debug('before JArray()')
        jrows = [ JArray(cast_func)(row) for row in ndarray ]
        _log.debug('after JArray()')

        jobjectTable = JArray('object')(jrows)
        _log.debug(type(jobjectTable))
        try:
            A = factory.matrix(jobjectTable)
        except AttributeError:
            A = factory.table(jobjectTable)
        
        _log.debug('finished setting values.')

        return A
    else:
        raise ValueError('unsupported shape: ', shape)


def stallone_array_to_ndarray(stArray):
    """
    Returns
    -------
    ndarray : 
    
    
    This subclass of numpy multidimensional array class aims to wrap array types
    from the Stallone library for easy mathematical operations.
    
    Currently it copies the memory, because the Python Java wrapper for arrays
    JArray<T> does not suggerate continuous memory layout, which is needed for
    direct wrapping.
    """
    # if first argument is of type IIntArray or IDoubleArray
    if not isinstance(stArray, (IIntArray, IDoubleArray)):
        raise TypeError('can only convert IDouble- or IIntArrays')
    
    from numpy import float64, int32, int64, array as nparray, empty
    dtype = None
    caster = None

    from platform import architecture
    arch = architecture()[0]
    if type(stArray) == IDoubleArray:
        dtype = float64
        caster = JArray_double.cast_
    elif type(stArray) == IIntArray:
        caster = JArray_int.cast_
        if arch == '64bit':
            dtype = int64 # long int?
        else:
            dtype = int32

    d_arr = stArray
    rows = d_arr.rows()
    cols = d_arr.columns()
    order = d_arr.order() 
    
    # TODO: support sparse
    #isSparse = d_arr.isSparse()
    
    if order < 2:
        #arr =  np.fromiter(d_arr.getArray(), dtype=dtype)
        # np.frombuffer(d_arr.getArray(), dtype=dtype, count=size )
        arr = nparray(d_arr.getArray(), dtype=dtype)
    elif order == 2:
        table = d_arr.getTable()
        arr = empty((rows, cols))
        # assign rows
        for i in xrange(rows):
            jarray = caster(table[i])
            row = nparray(jarray, dtype=dtype)
            arr[i] = row
    elif order == 3:
        raise NotImplemented
        
    arr.shape = (rows, cols)
    return arr
