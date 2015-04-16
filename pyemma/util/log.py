
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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