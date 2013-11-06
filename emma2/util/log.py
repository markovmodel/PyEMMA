'''
Created on 15.10.2013

@author: marscher
'''
import logging
from logging import _levelNames as loglevel_names
import args

import argparse

__all__ = ['log']

"""
    Arguments for logging
"""
parser = argparse.ArgumentParser(parents=[args.root_parser],
                                 add_help=False,
                                 description='Emma2 logging arguments')

parser.add_argument('--logging', dest='logging', default='on',
                    choices = ['on', 'off'],
                    help='turn logging on or off')
parser.add_argument('--console', action='store_true', default=True,
                     help='should be logged to console?')
parser.add_argument('--logfile', dest='logfilename', type=str,
                    default='emma2.log',
                    help='logfile to use')
parser.add_argument('--loglevel', dest='loglevel',
                    default=logging.INFO,
                    choices = ['NOTSET','DEBUG','INFO','WARNING',\
                               'ERROR','CRITICAL'],
                    help='minimum loglevel')

args = parser.parse_args()

#TODO: setup file logging here, but do not define a logger object and use
# import logging; log=logging.getLogger(__name__) in each unit instead.
# this should work, because everthing is derived from root logger
log = logging.getLogger('emma2')

if args.logging == 'on':
    # if console flag is on, ignore filename
    if args.console:
        _filename = None
    else:
        _filename = args.logfilename

    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M:%S',
                filename=_filename,
                filemode='a')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    _console = logging.StreamHandler()
    _console.setLevel(args.loglevel)
    # set a format which is simpler for console use
    _formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    _console.setFormatter(_formatter)
    
    log.addHandler(_console)
