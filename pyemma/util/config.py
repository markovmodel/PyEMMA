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
doc=r'''
Runtime Configuration
=====================
To configure the runtime behavior such as logging the system or other parameters,
the configuration module reads several config files to build
its final set of settings. It searches for the file 'pyemma.cfg' in several
locations with different priorities:

#. $CWD/pyemma.cfg
#. $HOME/.pyemma/pyemma.cfg
#. ~/pyemma.cfg
#. $PYTHONPATH/pyemma/pyemma.cfg (always taken as default configuration file)

The same applies for the filename ".pyemma.cfg" (hidden file).

The default values are stored in latter file to ensure these values are always
defined. This is preferred over hardcoding them somewhere in the Python code.

Default configuration file
--------------------------
Default settings are stored in a provided pyemma.cfg file, which is included in
the Python package:

.. literalinclude:: ../../pyemma/pyemma.cfg
    :language: ini

To access the config at runtime eg. the logging section:

>>> from pyemma import config
>>> print config.Logging.level # doctest: +SKIP
... INFO
>>> print config.show_progress_bars # doctest: +SKIP
... 'True'


Notes
-----
All values are being stored as strings, so to compare eg. if a value is True,
compare for:

.. code-block:: python

    if pyemma.config.show_progress_bars == 'True':
        ...

'''
import ConfigParser
import os
import sys
import warnings

from pyemma.util.files import mkdir_p

__docformat__ = "restructuredtext en"
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


def create_cfg_dir(default_config):
    home = os.path.expanduser("~")
    pyemma_cfg_dir = os.path.join(home, ".pyemma")
    if not os.path.exists(pyemma_cfg_dir):
        try:
            mkdir_p(pyemma_cfg_dir)
        except EnvironmentError:
            raise RuntimeError("could not create configuration directory '%s'" %
                               pyemma_cfg_dir)

    def touch(fname, times=None):
        with open(fname, 'a'):
            os.utime(fname, times)

    test_fn = os.path.join(pyemma_cfg_dir, "dummy")
    try:
        touch(test_fn)
        os.unlink(test_fn)
    except:
        raise RuntimeError("%s is not writeable" % pyemma_cfg_dir)

    # give user the default cfg file, if its not there
    import shutil
    if not os.path.exists(os.path.join(pyemma_cfg_dir, os.path.basename(default_config))):
        shutil.copy(default_config, pyemma_cfg_dir)

    return pyemma_cfg_dir


def readConfiguration():
    """
    TODO: consider using json to support arbitrary python objects in ini file (if this getting more complex)
    """
    import pkg_resources

    global configParser, conf_values, used_filenames

    cfg = 'pyemma.cfg'
    default_pyemma_conf = pkg_resources.resource_filename('pyemma', cfg)

    # create .pyemma dir in home
    pyemma_cfg_dir = ''
    try:
        pyemma_cfg_dir = create_cfg_dir(default_pyemma_conf)
    except RuntimeError as re:
        warnings.warn(str(re))

    # use these files to extend/overwrite the conf_values.
    # Last red file always overwrites existing values!
    filenames = [
        cfg,  # conf_values in current dir
        os.path.join(pyemma_cfg_dir, cfg),
        os.path.join(os.path.expanduser(
                     '~' + os.path.sep), cfg)  # config in user dir
    ]

    cfg_hidden = '.pyemma.cfg'
    filenames += [
        cfg_hidden,
        os.path.join(pyemma_cfg_dir, cfg),
        os.path.join(os.path.expanduser(
                     '~' + os.path.sep), cfg_hidden)
    ]

    # read defaults from default_pyemma_conf first.
    defParser = ConfigParser.RawConfigParser()

    try:
        with open(default_pyemma_conf) as f:
            defParser.readfp(f, default_pyemma_conf)
    except EnvironmentError as e:
        raise RuntimeError("FATAL ERROR: could not read default configuration"
                           " file %s\n%s" % (default_pyemma_conf, e))

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

    # remember cfg dir
    conf_values['pyemma']['cfg_dir'] = pyemma_cfg_dir

# read configuration once at import time
readConfiguration()


class Wrapper(object):

    __doc__ = doc

    # wrap attribute access for this module to enable shortcuts to config values
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        # try to lookup in conf_values first, then fall back to module attributes
        try:
            return conf_values['pyemma'][name]
        except KeyError:
            try:
                return conf_values[name]
            except KeyError:
                return getattr(self.wrapped, name)

    def __getitem__(self, name):
        try:
            return conf_values['pyemma'][name]
        except KeyError:
            return conf_values[name]

    def __setitem__(self, name, value):
        if name in conf_values['pyemma']:
            conf_values['pyemma'][name] = value
        elif name in conf_values:
            conf_values[name] = value
        else:
            raise KeyError('"%s" is not a valid config section.' % name)

sys.modules[__name__] = Wrapper(sys.modules[__name__])
