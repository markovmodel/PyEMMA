'''
Created on 15.10.2013

@author: marscher
'''
__all__ = ['getLogger', 'enabled', 'CRITICAL', 'DEBUG', 'FATAL', 'INFO', 'NOTSET',
           'WARN', 'WARNING']

import logging
reload(logging)

from logging import CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET

enabled = False


class dummyLogger:

    """ set up a dummy logger if logging is disabled"""

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

    enabled = args.enabled == 'True'
    toconsole = args.toconsole == 'True'
    tofile = args.tofile == 'True'

    if enabled:
        try:
            logging.basicConfig(level=args.level,
                                format=args.format,
                                datefmt='%d-%m-%y %H:%M:%S')
        except IOError as ie:
            import warnings
            warnings.warn(
                'logging could not be initialized, because of %s' % ie)
            return
        # in case we want to log to both file and stream, add a separate handler
        formatter = logging.Formatter(args.format)
        root_logger = logging.getLogger('')
        root_handlers = root_logger.handlers

        if toconsole:
            ch = root_handlers[0]
            ch.setLevel(args.level)
            ch.setFormatter(formatter)
        else: # remove first handler (which should be streamhandler)
            assert len(root_handlers) == 1
            streamhandler = root_handlers.pop()
            assert isinstance(streamhandler, logging.StreamHandler)
        if tofile:
            # set delay to True, to prevent creation of empty log files
            fh = logging.FileHandler(args.file, mode='a', delay=True)
            fh.setFormatter(formatter)
            fh.setLevel(args.level)
            root_logger.addHandler(fh)

        # if user enabled logging, but disallowed file and console logging, disable
        # logging completely.
        if not tofile and not toconsole:
            enabled = False
            dummyInstance = dummyLogger()
    else:
        dummyInstance = dummyLogger()


def getLogger(name=None):
    if not enabled:
        return dummyInstance
    # if name is not given, return a logger with name of the calling module.
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        path = t[0][0]
        pos = path.rfind('pyemma')
        if pos == -1:
            pos = path.rfind('scripts/')

        name = path[pos:]

    return logging.getLogger(name)


# init logging
setupLogging()
