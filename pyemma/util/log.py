'''
Created on 15.10.2013

@author: marscher
'''
__all__ = ['getLogger', 'enabled']

import logging

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
    parses pyemma configuration file and creates a logger conf_values from that
    """
    global enabled, dummyInstance
    from pyemma.util.config import conf_values
    args = conf_values['Logging']

    if args.enabled:
        try:
            logging.basicConfig(level=args.level,
                                format=args.format,
                                datefmt='%d-%m-%y %H:%M:%S')
        except IOError as ie:
            import warnings
            warnings.warn('logging could not be initialized, because of %s' % ie)
            return
        """ in case we want to log to both file and stream, add a separate handler"""
        formatter = logging.Formatter(args.format)

        if args.tofile:
            # set delay to True, to prevent creation of empty log files
            fh = logging.FileHandler(args.file, mode='a', delay=True)
            fh.setFormatter(formatter)
            fh.setLevel(args.level)
            logging.getLogger('').addHandler(fh)
        if args.toconsole and args.tofile:
            # since we have used basicConfig (without file args),
            # a StreamHandler is already used. Set our arguments.
            ch = logging.getLogger('').handlers[0]
            ch.setLevel(args.level)
            ch.setFormatter(formatter)
    else:
        dummyInstance = dummyLogger()

    enabled = args.enabled


def getLogger(name=None):
    if not enabled:
        return dummyInstance
    """ if name is not given, return a logger with name of the calling module."""
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        path = t[0][0]
        pos = path.rfind('pyemma')
        if pos == -1:
            pos = path.rfind('scripts/')

        name = path[pos:]

        #logging.getLogger().debug('getLogger set name to %s; path was %s' % (name, path))
    return logging.getLogger(name)


# init logging
setupLogging()