'''
Created on 15.10.2013

@author: marscher
'''
import logging
import ConfigParser

""" this filenames are being tried to read """
filenames = ['emma2.cfg', '/etc/emma2.cfg']
""" default values for logging system """
defaults = {'enabled': 'True',
            'toconsole' : 'True',
            'tofile' : 'False',
            'file' : 'emma2.log',
            'level' : 'DEBUG',
            'format' : '%%(asctime)s %%(name)-12s %%(levelname)-8s %%(message)s'}

class _AttribStore(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

config = ConfigParser.SafeConfigParser(defaults)
used_filenames = config.read(filenames)

if used_filenames == []:
    args = _AttribStore(defaults)
    """ we need to strip the string interpolation marks """
    args.format = args.format.replace('%%', '%')
else:
    section = 'Logging'
    args = _AttribStore()
    args.enabled = config.getboolean(section, 'enabled')
    args.toconsole = config.getboolean(section, 'toconsole')
    args.tofile = config.getboolean(section, 'tofile')
    args.file = config.get(section, 'file')
    args.level = config.get(section, 'level')
    args.format = config.get(section, 'format')
    
if args.enabled:
    if args.tofile and args.file:
        _filename = args.file
    else:
        _filename = None

    logging.basicConfig(level=args.level,
                format=args.format,
                datefmt='%d-%m-%y %H:%M:%S',
                filename=_filename,
                filemode='a')
    
    """ in case we want to log to both file and stream, add a separate handler"""
    if args.toconsole and args.tofile:
        ch = logging.StreamHandler()
        ch.setLevel(args.level)
        ch.setFormatter(logging.Formatter(args.format))
        logging.getLogger('').addHandler(ch)
        
log = logging.getLogger('emma2')