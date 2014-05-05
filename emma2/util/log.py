'''
Created on 15.10.2013

@author: marscher
'''
__all__ = ['getLogger', 'enabled']

import logging
from emma2.util.config import configParser as config, AttribStore

enabled = False

""" set up a dummy logger if logging is disabled"""
class dummyLogger:
    def dummy(self, kwargs):
        pass
    def __getattr__(self, name):
        return self.dummy
    
dummyInstance = None

def setupLogging():
    """
        parses emma2 configuration file and creates a logger config from that 
    """
    global enabled, dummyInstance
    
    section = 'Logging'
    args = AttribStore()
    args.enabled = config.getboolean(section, 'enabled')
    args.toconsole = config.getboolean(section, 'toconsole')
    args.tofile = config.getboolean(section, 'tofile')
    args.file = config.get(section, 'file')
    args.level = config.get(section, 'level')
    args.format = config.get(section, 'format')
    
    if args.enabled:
        if args.tofile and args.file:
            filename = args.file
        else:
            filename = None
        try:
            logging.basicConfig(level=args.level,
                    format=args.format,
                    datefmt='%d-%m-%y %H:%M:%S',
                    filename=filename,
                    filemode='a')
        except IOError as ie:
            import warnings
            warnings.warn('logging could not be initialized, because of %s' % ie)
            return
        """ in case we want to log to both file and stream, add a separate handler"""
        if args.toconsole and args.tofile:
            ch = logging.StreamHandler()
            ch.setLevel(args.level)
            ch.setFormatter(logging.Formatter(args.format))
            logging.getLogger('').addHandler(ch)
    else:
        dummyInstance = dummyLogger()
    
    enabled = args.enabled

def getLogger(name = None):
    if not enabled:
        return dummyInstance
    """ if name is not given, return a logger with name of the calling module."""
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        path = t[0][0]
        pos = path.rfind('emma2')
        name = path[pos:]
    return logging.getLogger(name)


# init logging
setupLogging()