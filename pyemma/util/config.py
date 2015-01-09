r'''
Runtime Configuration
=====================

To configure the runtime behavior like logging system or parameters for the
Java/Python bridge, the configuration module reads several config files to build
its final set of settings. It searches for the file 'pyemma.cfg' in several
locations with different priorities:

1. $CWD/pyemma.cfg
2. /etc/pyemma.cfg
3. ~/pyemma.cfg
4. $PYTHONPATH/Emma2/pyemma.cfg (always taken as default configuration file)

The default values are stored in later file to ensure these values are always
defined. This is preferred over hardcoding them somewhere in the Python code.


Default configuration file
--------------------------

Default settings are stored in a provided pyemma.cfg file, which is included in
the Python package:

.. literalinclude:: ../../pyemma/pyemma.cfg
    :language: ini

To access the config at runtime eg. the logging section 

.. code-block:: python

    from pyemma.util.config import config
    print config.Logging.level


.. codeauthor:: Martin Scherer <m.scherer at fu-berlin.de>

Members of module
-----------------

'''

__docformat__ = "restructuredtext en"

import ConfigParser
import os

__all__ = ['configParser', 'used_filenames', 'AttribStore']

configParser = None
"""instance of `ConfigParser.SafeConfigParser` to have always valid config values."""

used_filenames = []
"""these filenames have been tried to red to obtain basic configuration values."""

conf_values = None
"""holds all value pairs of a conf_values file section in a dict under its section name
eg. { 'Java' : { 'initheap' : '32m', ... }, ... }
"""


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
    import pkg_resources
    import sys

    global configParser, conf_values, used_filenames

    # use these files to extend/overwrite the conf_values.
    # Last red file always overwrites existing values!
    cfg = 'pyemma.cfg'
    filenames = [
        cfg,  # conf_values in current dir
        '/etc/' + cfg,  # conf_values in global installation
        os.path.join(os.path.expanduser(
                     '~' + os.path.sep), cfg),  # config in user dir
    ]

    # read defaults from default_pyemma_conf first.
    defParser = ConfigParser.RawConfigParser()
    default_pyemma_conf = pkg_resources.resource_filename('pyemma', cfg)
    try:
        with open(default_pyemma_conf) as f:
            defParser.readfp(f, default_pyemma_conf)
    except EnvironmentError as e:
        raise RuntimeError("FATAL ERROR: could not read default configuration"
                           " file %s\n%s" % (default_pyemma_conf, e))

    # handle case of different max heap sizes on 32/64 bit
    is64bit = sys.maxsize > 2 ** 32
    if is64bit:
        maxheap = 2000
    else:
        maxheap = 1280

    # if we have psutil, try to maximize memory usage.
    try:
        import psutil
        # available virtual memory in mb
        max_avail = psutil.virtual_memory().available / 1024**2

        maxheap = max(maxheap, max_avail)
    except ImportError:
        pass

    if maxheap < 1024:
        import warnings
        warnings.warn('Less than 1 GB of free memory. Underlying Java virtual'
                      ' machine capped to %s mb. Working with trajectories on'
                      ' Java side may cause memory problems.' % maxheap)

    defParser.set('Java', 'maxHeap', '%sm' % maxheap)

    # store values of defParser in configParser with sections
    configParser = ConfigParser.SafeConfigParser()
    for section in defParser.sections():
        configParser.add_section(section)
        for item in defParser.items(section):
            configParser.set(section, item[0], item[1])

    # this is a list of used configuration filenames during parsing the conf
    used_filenames = configParser.read(filenames)

    # store values in dictionaries for easy access
    conf_values = AttribStore()
    for section in configParser.sections():
        conf_values[section] = AttribStore()
        for item in configParser.items(section):
            conf_values[section][item[0]] = item[1]

readConfiguration()
