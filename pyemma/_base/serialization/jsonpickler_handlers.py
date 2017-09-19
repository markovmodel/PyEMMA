"""
This module contains custom serialization handlers for jsonpickle to flatten and restore given types.

@author: Martin K. Scherer
"""
import numpy as np

from jsonpickle import handlers


def register_ndarray_handler():
    H5BackendLinkageHandler.handles(np.ndarray)

    for t in NumpyExtractedDtypeHandler.np_dtypes:
        NumpyExtractedDtypeHandler.handles(t)

def register_all_handlers():
    register_ndarray_handler()


class H5BackendLinkageHandler(handlers.BaseHandler):
    """ stores NumPy arrays in the backing hdf5 file contained in the context """
    def __init__(self, context):
        if not hasattr(context, 'h5_file'):
            raise ValueError('the given un/-pickler has to contain a hdf5 file reference.')

        from jsonpickle.pickler import Pickler
        if isinstance(self, Pickler) and not hasattr(context, 'next_array_id'):
            raise ValueError('the given pickler has to contain an array id provider')
        super(H5BackendLinkageHandler, self).__init__(context=context)

    @property
    def file(self):
        # obtain the current file handler
        return self.context.h5_file

    def next_array_id(self):
        res = str(next(self.context.next_array_id))
        assert res not in self.file
        return res

    def flatten(self, obj, data):
        if obj.dtype == np.object_:
            value = [self.context.flatten(v, reset=False) for v in obj]
            data['values'] = value
        else:
            array_id = self.next_array_id()
            self.file.create_dataset(name=array_id, data=obj,
                                     chunks=True, compression='gzip', compression_opts=4, shuffle=True)
            data['array_ref'] = array_id
        return data

    def restore(self, obj):
        if 'array_ref' in obj:
            array_ref = obj['array_ref']
            # it is important to return a copy here, because h5 only creates views to the data.
            ds = self.file[array_ref]
            return ds[:]
        else:
            values = obj['values']
            result = np.empty(len(values), dtype=object)
            for i, e in enumerate(values):
                result[i] = self.context.restore(e, reset=False)
            return result


class NumpyExtractedDtypeHandler(handlers.BaseHandler):
    """
    if one extracts a value from a numpy array, the resulting type is numpy.int64 etc.
    We convert these values to Python primitives right here.

    All float types up to float64 are mapped by builtin.float
    All integer (signed/unsigned) types up to int64 are mapped by builtin.int
    """
    integers = (np.bool_,
                np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64)
    floats__ = (np.float16, np.float32, np.float64)

    np_dtypes = integers + floats__

    def __init__(self, context):
        super(NumpyExtractedDtypeHandler, self).__init__(context=context)

    def flatten(self, obj, data):
        if isinstance(obj, self.floats__):
            data['value'] = '{:.18f}'.format(obj).rstrip('0').rstrip('.')
        elif isinstance(obj, self.integers):
            data['value'] = int(obj)
        elif isinstance(obj, np.bool_):
            data['value'] = bool(obj)
        return data

    def restore(self, obj):
        str_t = obj['py/object'].split('.')[1]
        res = getattr(np, str_t)(obj['value'])
        return res
