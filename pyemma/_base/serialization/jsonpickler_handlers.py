"""
This module contains custom serialization handlers for jsonpickle to flatten and restore given types.

@author: Martin K. Scherer
"""

import numpy as np
from io import BytesIO

from pyemma._ext.jsonpickle import handlers
from pyemma._ext.jsonpickle import util
from pyemma._ext.jsonpickle.handlers import unregister as _unregister
from pyemma._ext.jsonpickle.ext.numpy import (register_handlers as _register_handlers,
                                              unregister_handlers as _unregister_handlers)


def register_ndarray_handler():
    """ Override jsonpickle handler for numpy arrays with compressed NPZ handler.
    First unregisters the default handler
    """
    _unregister_handlers()
    NumpyNPZHandler.handles(np.ndarray)


def unregister_ndarray_handler():
    """ Restore jsonpickle default numpy array handler.
    """
    _unregister(np.ndarray)
    _register_handlers()


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
        fh = np.load(buff)
        array = fh['x']
        fh.close()
        return array
