"""
This module contains custom serialization handlers for jsonpickle to flatten and restore given types.

@author: Martin K. Scherer
"""
import numpy as np

from pyemma._ext.jsonpickle import handlers


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
        super(H5BackendLinkageHandler, self).__init__(context=context)

    @property
    def file(self):
        # obtain the current file handler
        return self.context.h5_file

    def flatten(self, obj, data):
        if obj.dtype == np.object_:
            value = [self.context.flatten(v, reset=False) for v in obj]
            data['values'] = value
        else:
            import uuid
            array_id = '{group}/{id}'.format(group=self.file.name, id=uuid.uuid4())
            self.file.create_dataset(name=array_id, data=obj,
                                     chunks=True, compression='gzip', compression_opts=4, shuffle=True)
            data['array_ref'] = array_id
        return data

    def restore(self, obj):
        if 'array_ref' in obj:
            array_ref = obj['array_ref']
            # it is important to return a copy here, because h5 only creates views to the data.
            return self.file[array_ref][:]
        else:
            result = np.empty(len(obj), dtype=object)
            for i, e in enumerate(obj['values']):
                result[i] = self.restore(e)
            return result


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
