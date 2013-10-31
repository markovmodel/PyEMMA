'''
Created on 15.10.2013

@author: marscher
'''
import logging
import os
import emma2 as _emma2

# import log levels to this namespace
"""
    global logger for emma2 application
    
    log.debug('debug message')
    log.info('info message')
    log.warn('warn message')
    log.error('error message')
    log.critical('critical message')
"""
#TODO: setup file logging here, but do not define a logger object and use
# import logging; log=logging.getLogger(__name__) in each unit instead.
# this should work, because everthing is derived from root logger
log = logging.getLogger()

# set up logging to file - see previous section for more details
# TODO: martin prepend the path of emma2 module (+ '..') to filename
# and think about putting these options to an ini file etc.
_filename = os.getcwd() + '/emma2.log'
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M:%S',
            filename=_filename,
            filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
# set a format which is simpler for console use
_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
_console.setFormatter(_formatter)

log.addHandler(_console)