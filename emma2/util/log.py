'''
Created on 15.10.2013

@author: marscher
'''
import logging
import ConfigParser

class _AttribStore(dict):

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

defaults = {'enabled': True,
            'console' : True,
            'tofile' : False,
            'file' : 'emma2.log',
            'level' : 'DEBUG'}

config = ConfigParser.SafeConfigParser(defaults)
""" this filenames are being tried to read """
filenames = ['emma2.cfg', '/etc/emma2.cfg']
used_filenames = config.read(filenames)

section = 'Logging'
args = _AttribStore()
args.logging = config.getboolean(section, 'enabled')
args.console = config.getboolean(section, 'toconsole')
args.tofile = config.getboolean(section, 'tofile')
args.logfilename = config.get(section, 'file')
args.loglevel = config.get(section, 'level')

#TODO: setup file logging here, but do not define a logger object and use
# import logging; log=logging.getLogger(__name__) in each unit instead.
# this should work, because everthing is derived from root logger
log = logging.getLogger('emma2')

if args.logging:
    # if console flag is on, ignore filename
    if args.console:
        _filename = None
    else:
        _filename = args.logfilename

    logging.basicConfig(level=args.loglevel,
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%d-%m-%y %H:%M:%S',
                filename=_filename,
                filemode='a')
