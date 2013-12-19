'''
Created on 15.10.2013

@author: marscher
'''
__all__ = ['log', 'getLogger', 'logargs']

import logging
import ConfigParser
import os

""" this filenames are being tried to read to obtain basic configuration values 
    for the logging system."""
cfg = 'emma2.cfg'
filenames = [cfg,
            '/etc/' + cfg,
            os.path.expanduser('~') + cfg,
            # This should always be last
            os.path.dirname(__import__('emma2').__file__) + cfg,
            ]
""" default values for logging system """
defaults = {'enabled': 'True',
            'toconsole' : 'True',
            'tofile' : 'False',
            'file' : 'emma2.log',
            'level' : 'DEBUG',
            'format' : '%%(asctime)s %%(name)-12s %%(levelname)-8s %%(message)s'}

class AttribStore(dict):
    def __getattr__(self, name):
        return self[name]
 
    def __setattr__(self, name, value):
        self[name] = value

config = ConfigParser.SafeConfigParser(defaults)
used_filenames = config.read(filenames)

if used_filenames == []:
    args = AttribStore(defaults)
    """ we need to strip the string interpolation marks """
    args.format = args.format.replace('%%', '%')
else:
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

    logging.basicConfig(level=args.level,
                format=args.format,
                datefmt='%d-%m-%y %H:%M:%S',
                filename=filename,
                filemode='a')
    
    """ in case we want to log to both file and stream, add a separate handler"""
    if args.toconsole and args.tofile:
        ch = logging.StreamHandler()
        ch.setLevel(args.level)
        ch.setFormatter(logging.Formatter(args.format))
        logging.getLogger('').addHandler(ch)

    """ default logger for emma2 """
    log = logging.getLogger('emma2')
else:
    """ set up a dummy logger if logging is disabled"""
    class dummyLogger:
        def log(self, kwargs):
            pass
        def debug(self, kwargs):
            pass
        def info(self, kwargs):
            pass
        def warn(self, kwargs):
            pass
        def error(self, kwargs):
            pass
        def critical(self, kwargs):
            pass
        def setLevel(self, kwargs):
            pass
    log = dummyLogger()

def getLogger(name = None):
    if not args.enabled:
        return dummyLogger()
    """ if name is not given, return a logger with name of the calling module."""
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        path = t[0][0]
        pos = path.rfind('emma2')
        name = path[pos:]
    return logging.getLogger(name)

def logargs(func):
    """
    use like this:
    >>> @logargs
    >>> def sample():
    >>>    return 2
    >>> sample(1, 3)
    Arguments to function sample were: (1, 3), {}
    
    """
    def inner(*args, **kwargs): #1
        log.debug("Arguments to function %s were: %s, %s" 
                  % (func.__name__, args, kwargs))
        return func(*args, **kwargs) #2
    return inner