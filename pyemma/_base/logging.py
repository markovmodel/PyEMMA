'''
Created on 30.08.2015

@author: marscher
'''
import logging
import weakref
from pyemma.util.log import getLogger

__all__ = ['create_logger', 'instance_name']

_refs = {}

def _cleanup_logger(obj):
    # callback function used in conjunction with weakref.ref to remove logger for transformer instance
    key = obj._logger_instance.name

    def remove_logger(weak):
        d = logging.getLogger().manager.loggerDict
        del d[key]
        del _refs[key]
    return remove_logger

def _clean_dead_refs():
    # cleans dead weakrefs
    global _refs
     
    if len(_refs) == 0:
        return
    _refs = [r for r in _refs if r() is not None]
    
def instance_name(self, id):
    package = self.__module__
    instance_name = "%s.%s[%i]" % (package, self.__class__.__name__, id)
    return instance_name

def create_logger(self):
    # creates a logger based on the the attribe "name" of self
    self._logger_instance = getLogger(self.name)
    r = weakref.ref(self, _cleanup_logger(self))
    _refs[self.name] = r
    return self._logger_instance
