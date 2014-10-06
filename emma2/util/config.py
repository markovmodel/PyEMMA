r'''
Runtime Configuration
=====================

To configure the runtime behavior like logging system or parameters for the
Java/Python bridge, the configuration module reads several config files to build
its final set of settings. It searches for the file 'emma2.cfg' in several
locations with different priorities:

1. $CWD/emma2.cfg
2. /etc/emma2.cfg
3. ~/emma2.cfg
4. $PYTHONPATH/Emma2/emma2.cfg (always taken as default configuration file)

The default values are stored in later file to ensure these values are always
defined. This is preferred over hardcoding them somewhere in the Python code.


Default configuration file
--------------------------

Default settings are stored in a provided emma2.cfg file, which is included in
the Python package:

.. literalinclude:: ../../emma2/emma2.cfg
    :language: ini


.. codeauthor:: Martin Scherer <m.scherer at fu-berlin.de>

Members of module
-----------------

'''

__docformat__ = "restructuredtext en"

import ConfigParser
import os
import pkg_resources

__all__ = ['configParser', 'used_filenames', 'AttribStore']

configParser = None
""" instance of `ConfigParser.SafeConfigParser` to have always valid config values."""

used_filenames = []
""" these filenames have been tried to red to obtain basic configuration values."""

class AttribStore(dict):
    """ store arbitrary attributes in this dictionary like class."""
    def __getattr__(self, name):
        """ return attribute with given name or raise."""
        return self[name]

    def __setattr__(self, name, value):
        """ store attribute with given name and value."""
        self[name] = value

def readConfiguration():
    """
    TODO: consider using json to support arbitray python objects in ini file (if this getting more complex)
    """
    
    global configParser, used_filenames
        
    # use these files to extend/overwrite the config.
    # Last red file always overwrites existing values!
    cfg = 'emma2.cfg'
    filenames = [cfg, # config in current dir
                '/etc/' + cfg, # config in global installation
                os.path.join(os.path.expanduser('~' + os.path.sep), cfg), # config in user dir
                ]
    
    # read defaults from default_emma2_conf first.
    defParser = ConfigParser.RawConfigParser()
    default_emma2_conf = pkg_resources.resource_filename('emma2', cfg)
    try:
        with open(default_emma2_conf) as f:
            defParser.readfp(f, default_emma2_conf)
    except EnvironmentError as e:
        raise RuntimeError("FATAL ERROR: could not read default configuration file %s\n%s"
              % (default_emma2_conf, e))
    
    # store values of defParser in configParser with sections
    configParser = ConfigParser.SafeConfigParser()
    for section in defParser.sections():
        configParser.add_section(section)
        for item in defParser.items(section):
            configParser.set(section, item[0], item[1])
            
    """ this is a list of used configuration filenames during parsing the configuration"""
    used_filenames = configParser.read(filenames)

readConfiguration()