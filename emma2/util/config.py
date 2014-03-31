'''
The Emma2 configuration module reads several config files to build its final
set of settings. It searches for the file 'emma2.cfg' in several locations with
different priorities:
1. $CWD/emma2.cfg
2. /etc/emma2.cfg
3. ~/emma2.cfg
4. $PYTHONPATH/Emma2/emma2.cfg

The default values are stored in later file to ensure these values are always
defined. This is preferred over hardcoding them somewhere in the python code.

Created on 31.03.2014

@author: marscher
'''
import ConfigParser
import os
import pkg_resources

__all__ = ['configParser', 'used_filenames', 'AttribStore']

""" this filenames are being tried to read to obtain basic configuration values 
    for the logging system."""
cfg = 'emma2.cfg'

ultimate_backup = pkg_resources.resource_filename('emma2', os.path.join('..', 'emma2.cfg'))
filenames = [cfg, # config in current dir
            '/etc/' + cfg, # config in global installation
            os.path.join(os.path.expanduser('~'), os.path.sep, cfg), # config in user dir
            # This should always be last, since it provides all default values
            # and things will horribly fail, if this file can not be found.
            ultimate_backup
            ]

class AttribStore(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

configParser = ConfigParser.SafeConfigParser()
""" this is a list of used configuration filenames during parsing the configuration"""
used_filenames = configParser.read(filenames)

if ultimate_backup not in used_filenames:
    raise RuntimeError('Default configuration values could not be red!')