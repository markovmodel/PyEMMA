'''
This module is used to initialize a global Java VM instance, to run the python
wrapper for the stallone library
Created on 15.10.2013

@author: marscher
'''
from log import log

"""is the stallone python binding available?"""
stallone_available = None
jenv = None
try:
    log.debug('try to initialize stallone module')
    import pystallone as stallone
    from pystallone.ArrayWrapper import ArrayWrapper as _ArrayWrapper
    # add ArrayWrapper to stallone module 
    stallone.ArrayWrapper = _ArrayWrapper
    jenv = stallone.initVM(initialheap='32m', maxheap='512m')
    stallone_available = True
    log.debug('stallone initialized successfully.')
except ImportError:
    log.error('stallone could not be found.')
    stallone_available = False
except ValueError as ve:
    stallone_available = False
    log.error('java vm initialization for stallone went wrong: ', ve)
except:
    log.error('unknown exception occured.')