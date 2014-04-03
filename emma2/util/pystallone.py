'''
This module is used to initialize a global Java VM instance, to run the Python
wrapper for the Stallone library.

The API variable is the main entry point into the Stallone API factory.


Examples
--------
create a double vector and assigning values:
>>> from emma2.util.pystallone import stallone as st
>>> x = st.api.API.doublesNew.array(10) # create double array with 10 elements
>>> x.set(5, 23.0) # set index 5 to 23.0
>>> print(x)
0.0     0.0     0.0     0.0     0.0     23.0     0.0     0.0     0.0     0.0

Created on 15.10.2013

@author: marscher
'''
from log import getLogger as _getLogger
_log = _getLogger(__name__)

from jpype import \
 startJVM as _startJVM, \
 getDefaultJVMPath as _getDefaultJVMPath, \
 JavaException, \
 JArray, JInt, JDouble, JString, JObject, JPackage, \
 java, javax

import numpy as _np
import sys as _sys

_64bit = _sys.maxsize > 2**32

""" stallone java package. Should be used to access all classes in the stallone library."""
stallone = None
""" main stallone API entry point """
API = None

def _initVM():
    import os
    import pkg_resources
    from emma2.util.config import configParser
    
    global stallone, API
    
    def buildClassPath():
        # define classpath separator
        if os.name is 'posix':
            sep = ':'
        else:
            sep = ';'
        
        stallone_jar = os.path.join('..','lib','stallone',
                                    'stallone-1.0-SNAPSHOT-jar-with-dependencies.jar')
        stallone_jar_file = pkg_resources.resource_filename('emma2', stallone_jar)
        if not os.path.exists(stallone_jar_file):
            raise RuntimeError('stallone jar not found! Expected it here: %s' 
                           % stallone_jar_file)
        # cleanup optional cp (fix separator char etc)
        optional_cp = configParser.get('Java', 'classpath')
        if os.name is 'posix':
            optional_cp.replace(';', sep)
        else:
            optional_cp.replace(':', sep)
            
        # warn user about non existing custom cp
        cp = []
        for p in optional_cp.split(sep):
            if p is not '' and not os.path.exists(p):
                _log.warning('custom classpath "%s" does not exist!' % p)
            else:
                cp.append(p)

        # user classpaths first, then stallone jar (to overwrite it optionally) 
        cp.append(stallone_jar_file)
        return '-Djava.class.path=' + sep.join(cp)
    
    classpath = buildClassPath()
    initHeap = '-Xms%s' % configParser.get('Java', 'initHeap')
    maxHeap = '-Xmx%s' % configParser.get('Java', 'maxHeap')
    optionalArgs = configParser.get('Java', 'optionalArgs')
    
    args = [initHeap, maxHeap, classpath, optionalArgs]
    
    try:
        _log.debug('init with options: "%s"' % args)
        _log.debug('default vm path: %s' % _getDefaultJVMPath())
        _startJVM(_getDefaultJVMPath(), *args)
    except RuntimeError:
        _log.exception('startJVM failed.')
        raise

    try:
        stallone = JPackage('stallone')
        API = stallone.api.API
        if type(API).__name__ != 'stallone.api.API$$Static':
            raise RuntimeError('Stallone package initialization borked. Check your JAR/classpath!') 
    except Exception:
        _log.exception('initialization went wrong')
        raise

_initVM()

def ndarray_to_stallone_array(pyarray):
    """
        Parameters
        ----------
        pyarray : numpy.ndarray or scipy.sparse type one or two dimensional
        
        Returns
        -------
        IDoubleArray or IIntArray depending on input type
        
        Note:
        -----
        scipy.sparse types will be currently converted to dense, before passing
        them to the java side!
    """
    from scipy.sparse.base import issparse
    if issparse(pyarray):
        _log.warning("converting sparse object to dense for stallone.")
        pyarray = pyarray.todense()
    
    shape = pyarray.shape
    dtype = pyarray.dtype
    factory = None
    cast_func = None
    
    if dtype == _np.float32 or dtype == _np.float64:
        factory = API.doublesNew
        cast_func = JDouble
    elif dtype == _np.int32 or dtype == _np.int64:
        factory = API.intsNew
        cast_func = JInt
    else:
        raise TypeError('unsupported datatype:', dtype)

    if len(shape) == 1:
        # create a JArray wrapper
        jarr = JArray(cast_func)(pyarray)
        if cast_func is JDouble:
            return factory.array(jarr)
        if cast_func is JInt:
            return factory.arrayFrom(jarr)
        raise TypeError('type not mapped to a stallone factory')

    elif len(shape) == 2:
        # TODO: use linear memory layout here, when supported in stallone
        jarr = JArray(cast_func, 2)(pyarray)
        try:
            # for double arrays
            A = factory.array(jarr)
        except AttributeError:
            # for int 2d arrays
            A = factory.table(jarr)
        return A
    else:
        raise ValueError('unsupported shape:', shape)



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
    if not isinstance(stArray, (stallone.api.ints.IIntArray,
                                stallone.api.doubles.IDoubleArray)):
        raise TypeError('can only convert IDouble- or IIntArrays')
    
    dtype = None

    if type(stArray) == stallone.api.doubles.IDoubleArray:
        dtype = _np.float64
    elif type(stArray) == stallone.api.ints.IIntArray:
        if _64bit:
            dtype = _np.int64
        else:
            dtype = _np.int32

    d_arr = stArray
    rows = d_arr.rows()
    cols = d_arr.columns()
    order = d_arr.order() 
    
    # TODO: support sparse
    # isSparse = d_arr.isSparse()
    
    # construct an ndarray using a slice onto the JArray
    # make sure to always use slices, if you want to access ranges (0:n), else
    # an getter for every single element will be called and you can you will wait for ages.
    if order < 2:
        jarr = d_arr.getArray()
        arr = _np.array(jarr[0:len(jarr)], dtype=dtype, copy=False)
    elif order == 2:
        jarr = d_arr.getArray()
        arr = _np.array(jarr[0:len(jarr)], dtype=dtype, copy=False)
        #arr = _np.array(jarr[0:len(jarr)], copy=False)
        #arr = _np.zeros((rows,cols))
    else:
        raise NotImplemented
    
    if cols > 1:
        shape = (rows, cols)
    else:
        shape = (rows,)

    return arr.reshape(shape)


def list1d_to_java_array(a):
    """
    Converts python list of primitive int or double to java array
    """
    if type(a) is list:
        if type(a[0]) is int:
            return JArray(JInt)(a)
        elif type(a[0]) is float:
            return JArray(JDouble)(a)
        elif type(a[0]) is str:
            return JArray(JString)(a)
        else:
            return JArray(JObject)(a)
    else:
        raise TypeError("Not a list: " + str(a))

def list_to_java_list(a):
    """
    Converts python list of primitive int or double to java array
    """
    if type(a) is list:
        jlist = java.util.ArrayList()
        for el in a:
            jlist.add(el)
        return jlist
    else:
        raise TypeError("Not a list: " + str(a))


def list2d_to_java_array(a):
    """
    Converts python list of primitive int or double to java array
    """
    if type(a) is list:
        if type(a[0]) is list:
            if type(a[0][0]) is int:
                return JArray(JInt,2)(a)
            elif type(a[0][0]) is float:
                return JArray(JDouble,2)(a)
            elif type(a[0][0]) is str:
                return JArray(JString,2)(a)
            else:
                return JArray(JObject,2)(a)
        else:
            raise TypeError("Not a list: " + str(a[0]))
    else:
        raise TypeError("Not a list: " + str(a))


def list_to_jarray(a):
    """
    Converts 1d or 2d python list of primitive int or double to
    java array or nested array
    """
    if type(a) is list:
        if type(a[0]) is list:
            return list2d_to_java_array(a)
        else:
            return list1d_to_java_array(a)


def jarray(a):
    """
    Converts array-like (python list or ndarray) to java array
    """
    if type(a) is list:
        return list_to_jarray(a)
    elif isinstance(a, _np.ndarray):
        return list_to_jarray(a.tolist())
    else:
        raise TypeError("Type is not supported for conversion to java array")
