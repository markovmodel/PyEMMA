'''
Created on 15.10.2013

@author: marscher
'''
from logging import DEBUG, INFO, WARN, ERROR, CRITICAL
import logging

# import log levels to this namespace
"""
    global logger for emma2 application
    
    log.debug('debug message')
    log.info('info message')
    log.warn('warn message')
    log.error('error message')
    log.critical('critical message')
"""
log = logging.getLogger('emma2')

# set up logging to file - see previous section for more details
logging.basicConfig(level=DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename='emma2.log',
            filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
_console = logging.StreamHandler()
_console.setLevel(INFO)
# set a format which is simpler for console use
_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
_console.setFormatter(_formatter)

log.addHandler(_console)