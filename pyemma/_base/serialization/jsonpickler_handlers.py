"""
This module contains custom serialization handlers for jsonpickle to flatten and restore given types.

@author: Martin K. Scherer
"""

from io import BytesIO

import numpy as np

from pyemma._ext.jsonpickle import handlers
from pyemma._ext.jsonpickle import util


def register_ndarray_handler():
    """ Override jsonpickle handler for numpy arrays with compressed NPZ handler.
    First unregisters the default handler
    """
    NumpyNPZHandler.handles(np.ndarray)

    for t in NumpyExtractedDtypeHandler.np_dtypes:
        NumpyExtractedDtypeHandler.handles(t)


class NumpyNPZHandler(handlers.BaseHandler):
    """ stores NumPy array as a compressed NPZ file. """
    def __init__(self, context):
        super(NumpyNPZHandler, self).__init__(context=context)

    def flatten(self, obj, data):
        assert isinstance(obj, np.ndarray)
        buff = BytesIO()
        np.savez_compressed(buff, x=obj)
        buff.seek(0)
        flattened_bytes = util.b64encode(buff.read())
        data['npz_file_bytes'] = flattened_bytes
        return data

    def restore(self, obj):
        binary = util.b64decode(obj['npz_file_bytes'])
        buff = BytesIO(binary)
        with np.load(buff) as fh:
            array = fh['x']
        return array


class NumpyExtractedDtypeHandler(handlers.BaseHandler):
    """
    if one extracts a value from a numpy array, the resulting type is numpy.int64 etc.
    We convert these values to Python primitives right here.
    
    All float types up to float64 are mapped by builtin.float
    All integer (signed/unsigned) types up to int64 are mapped by builtin.int
    """
    np_dtypes = (np.float16, np.float32, np.float64,
                 np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64)

    def __init__(self, context):
        super(NumpyExtractedDtypeHandler, self).__init__(context=context)

    def flatten(self, obj, data):
        str_t = type(obj).__name__
        if str_t.startswith("float"):
            value = float(obj)
        elif str_t.startswith("int") or str_t.startswith("uint"):
            value = int(obj)
        else:
            raise ValueError("not yet impled for type %s" % str_t)

        return value

    def restore(self, obj):
        raise RuntimeError("this should never be called, because the handled types are converted to primitives.")
