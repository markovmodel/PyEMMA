'''
This module is used to initialize a global Java VM instance, to run the Python
wrapper for the Stallone library.

The API variable is the main entry point into the Stallone API factory.


Examples
--------
Read a trajectory:
>>> from emma2.util.pystallone import stallone as st
>>> st.api.API.data.

Created on 15.10.2013

@author: marscher
'''
from log import getLogger
from scipy.sparse.base import issparse
_log = getLogger(__name__)
# need this for ipython!!!
#_log.setLevel(50)

from jpype import \
 startJVM as _startJVM, \
 getDefaultJVMPath as _getDefaultJVMPath, \
 JavaException, \
 JArray, JInt, JDouble, JObject, JPackage

import numpy as _np

""" stallone java package. Should be used to access all classes in the stallone library."""
stallone = None
""" main stallone API entry point """
API = None

def _initVM():
    import os
    
    def getStalloneJarFilename():
        filename = 'stallone-1.0-SNAPSHOT-jar-with-dependencies.jar'
        abspath = os.path.abspath(__file__)
        abspath = os.path.dirname(abspath) + os.path.sep
        relpath = '../../lib/stallone/'.replace('/', os.path.sep)
        abspath = abspath + relpath + filename
        return abspath
    
    stallone_jar = getStalloneJarFilename()
    if not os.path.exists(stallone_jar):
        raise RuntimeError('stallone jar not found! Expected it here: %s' 
                           % stallone_jar)
    
    # TODO: store and read options in emma2.cfg
    classpath = "-Djava.class.path=%s%s" % (stallone_jar, os.sep)
    initHeap = "-Xms64m"
    maxHeap = "-Xms512m"
    
    args = [initHeap, maxHeap, classpath]
    try:
        _log.debug('init with options: "%s"' % args)
        _startJVM(_getDefaultJVMPath(), *args)
    except RuntimeError as re:
        _log.error(re)
        raise
    global stallone, API
    try:
        stallone = JPackage('stallone')
        API = stallone.api.API
        from jpype._jpackage import JPackage as jp
        if type(API) == type(jp): #TODO: this check does not work
            raise RuntimeError('jvm initialization borked. Type of API should be JClass')
    except Exception as e:
        _log.error(e)
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
        # TODO: remove this, when that is solved: https://github.com/originell/jpype/issues/24
        #pyarray=pyarray.astype(_np.int64)
        pyarray = pyarray.tolist() # this works always, but is undesired
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

    from platform import architecture
    arch = architecture()[0]
    if type(stArray) == stallone.api.doubles.IDoubleArray:
        dtype = _np.float64
    elif type(stArray) == stallone.api.ints.IIntArray:
        if arch == '64bit':
            dtype = _np.int64  # long int?
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
        else:
            return JArray(JDouble)(a)
    else:
        raise TypeError("Not a list: " + str(a))


def list_to_jarray(a):
    """
    Converts 1d or 2d python list of primitive int or double to
    java array or nested array
    """
    if type(a) is list:
        if type(a[0]) is int or type(a[0]) is float:
            return list1d_to_java_array(a)
        if type(a[0]) is list:
            n = len(a)
            ja = JArray(JObject)(n)
            for i in range(n):
                ja[i] = list1d_to_java_array(a[i])
            return ja


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
