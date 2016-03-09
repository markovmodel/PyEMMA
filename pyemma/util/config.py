
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

import os
import shutil
import sys
import warnings

from pyemma.util.annotators import deprecated
from pyemma.util.files import mkdir_p

import pkg_resources


# for IDE stupidity, just add a new cfg var here, if you add a property to Wrapper
cfg_dir = default_config_file = default_logging_config = logging_config = \
    show_progress_bars = used_filenames = use_trajectory_lengths_cache = None

__all__ = (
           'cfg_dir',
           'default_config_file',
           'default_logging_file',
           'logging_config',
           'show_progress_bars',
           'used_filenames',
           'use_trajectory_lengths_cache',
           )


class NotADirectoryError(Exception):
    pass


class Wrapper(object):
    r'''
Runtime Configuration
=====================
To configure the runtime behavior such as the logging system or other parameters,
the configuration module reads several config files to build
its final set of settings. It searches for the file 'pyemma.cfg' in several
locations with different priorities:

#. $CWD/pyemma.cfg
#. $HOME/.pyemma/pyemma.cfg
#. ~/pyemma.cfg
#. $PYTHONPATH/pyemma/pyemma.cfg (always taken as default configuration file)

The same applies for the filename ".pyemma.cfg" (hidden file).

Note that you can also override the location of the configuration directory by
setting an environment variable named "PYEMMA_CFG_DIR" to a writeable path to 
override the location of the config files.

The default values are stored in latter file to ensure these values are always
defined. This is preferred over hardcoding them somewhere in the Python code.

After the first import of pyemma, you will find a .pyemma directory in your
user directory. It contains a pyemma.cfg and logging.yml. The latter is a YAML
file to configure the logging system.
For details have a look at the brief documentation:
https://docs.python.org/2/howto/logging.html

Default configuration file
--------------------------
Default settings are stored in a provided pyemma.cfg file, which is included in
the Python package:

.. literalinclude:: ../../pyemma/pyemma.cfg
    :language: ini

To access the config at runtime eg. the logging section:

>>> from pyemma import config # doctest: +SKIP
>>> print(config.show_progress_bars) # doctest: +SKIP
True

or

>>> config.show_progress_bars = False # doctest: +SKIP
>>> print(config.show_progress_bars) # doctest: +SKIP
False
    '''
    DEFAULT_CONFIG_FILE_NAME = 'pyemma.cfg'
    DEFAULT_LOGGING_FILE_NAME = 'logging.yml'

    __name__ = 'pyemma.util.config'
    __file__ = __file__

    def __init__(self, wrapped):
        # create .pyemma dir in home
        try:
            self._create_cfg_dir()
        except RuntimeError as re:
            warnings.warn(str(re))

        try:
            self.__readConfiguration()
        except RuntimeError as re:
            warnings.warn("unable to read default configuration file. Logging and "
                          " progress bar handling could behave bad! Error: %s" % re)

        from pyemma.util.log import setupLogging, LoggingConfigurationError
        try:
            setupLogging(self)
        except LoggingConfigurationError as e:
            warnings.warn("Error during logging configuration. Logging might not be functional!"
                          "Error: %s" % e)

        # wrap this module
        self.wrapped = wrapped
        self.__wrapped__ = wrapped

    def __call__(self, ):
        return Wrapper(sys.modules[__name__])

    @property
    def cfg_dir(self):
        """ configuration directory (eg. in ~/.pyemma """
        if self._cfg_dir is None:
            self._cfg_dir = self._create_cfg_dir()
        return self._cfg_dir

    @property
    def used_filenames(self):
        """these filenames have been tried to red to obtain basic configuration values."""
        return self._used_filenames

    @property
    def default_config_file(self):
        """ default config file living in PyEMMA package """
        return pkg_resources.resource_filename('pyemma', Wrapper.DEFAULT_CONFIG_FILE_NAME)

    @property
    def default_logging_file(self):
        return pkg_resources.resource_filename('pyemma', Wrapper.DEFAULT_LOGGING_FILE_NAME)

    @deprecated("do not use this!")
    def conf_values(self):
        return self._conf_values

    def keys(self):
        return ['show_progress_bars',
                'use_trajectory_lengths_cache',
                'logging_config',
                ]

    ### SETTINGS
    @property
    def logging_config(self):
        cfg = self._conf_values['pyemma']['logging_config']
        if cfg == 'DEFAULT':
            cfg = os.path.join(self.cfg_dir, Wrapper.DEFAULT_LOGGING_FILE_NAME)
        return cfg

    @property
    def show_progress_bars(self):
        return bool(self._conf_values['pyemma']['show_progress_bars'])

    @show_progress_bars.setter
    def show_progress_bars(self, val):
        self._conf_values['pyemma']['show_progress_bars'] = bool(val)

    @property
    def use_trajectory_lengths_cache(self):
        return bool(self._conf_values['pyemma']['use_trajectory_lengths_cache'])

    @use_trajectory_lengths_cache.setter
    def use_trajectory_lengths_cache(self, val):
        self._conf_values['pyemma']['use_trajectory_lengths_cache'] = bool(val)

    def _create_cfg_dir(self):
        try:
            os.stat(self.default_config_file)
        except OSError:
            raise RuntimeError('Error during accessing default config file "%s"' %
                               self.default_config_file)

        if 'PYEMMA_CFG_DIR' in os.environ:
            pyemma_cfg_dir = os.environ['PYEMMA_CFG_DIR']
        else:
            pyemma_cfg_dir = os.path.join(os.path.expanduser("~"), ".pyemma")

        self._cfg_dir = pyemma_cfg_dir
        if not os.path.exists(pyemma_cfg_dir):
            try:
                mkdir_p(pyemma_cfg_dir)
            except EnvironmentError:
                raise RuntimeError("could not create configuration directory '%s'" %
                                   pyemma_cfg_dir)
            except NotADirectoryError:
                raise RuntimeWarning("pyemma cfg dir (%s) is not a directory" %
                                     pyemma_cfg_dir)

        if not os.path.isdir(pyemma_cfg_dir):
            raise RuntimeError("%s is no valid directory" % pyemma_cfg_dir)
        if not os.access(pyemma_cfg_dir, os.W_OK):
            raise RuntimeError("%s is not writeable" % pyemma_cfg_dir)

        # give user the default cfg file, if its not there
        files_to_check_copy = [
               Wrapper.DEFAULT_CONFIG_FILE_NAME,
               Wrapper.DEFAULT_LOGGING_FILE_NAME,
        ]
        dests = [os.path.join(pyemma_cfg_dir, f) for f in files_to_check_copy]
        srcs = [pkg_resources.resource_filename('pyemma', f) for f in files_to_check_copy]
        for src, dest in zip(srcs, dests):
            if not os.path.exists(dest):
                shutil.copyfile(src, dest)

    def __readConfiguration(self):
        """
        reads config files from various locations to build final config.
        """
        from six.moves import configparser

        # use these files to extend/overwrite the conf_values.
        # Last red file always overwrites existing values!
        cfg = Wrapper.DEFAULT_CONFIG_FILE_NAME
        filenames = [
            cfg,  # conf_values in current dir
            os.path.join(self.cfg_dir, cfg),
            os.path.join(os.path.expanduser(
                         '~' + os.path.sep), cfg)  # config in user dir
        ]

        cfg_hidden = '.pyemma.cfg'
        filenames += [
            cfg_hidden,
            # deprecated:
            os.path.join(self.cfg_dir, cfg_hidden),
            os.path.join(os.path.expanduser(
                         '~' + os.path.sep), cfg_hidden)
        ]

        # read defaults from default_pyemma_conf first.
        defParser = configparser.RawConfigParser()

        try:
            defParser.read(self.default_config_file)
        except EnvironmentError as e:
            raise RuntimeError("FATAL ERROR: could not read default configuration"
                               " file %s\n%s" % (self.default_config_file, e))

        # store values of defParser in configParser with sections
        configParser = configparser.SafeConfigParser()

        for section in defParser.sections():
            configParser.add_section(section)
            for item in defParser.items(section):
                configParser.set(section, item[0], item[1])

        # this is a list of used configuration filenames during parsing the
        # conf
        self._used_filenames = configParser.read(filenames)

        # store values in dictionaries for easy access
        conf_values = dict()
        for section in configParser.sections():
            conf_values[section] = dict()
            for item in configParser.items(section):
                conf_values[section][item[0]] = item[1]

        conf_values['pyemma']['cfg_dir'] = self.cfg_dir
        self._conf_values = conf_values

    # for dictionary like lookups
    def __getitem__(self, name):
        try:
            return self._conf_values['pyemma'][name]
        except KeyError:
            return self._conf_values[name]

    def __setitem__(self, name, value):
        self._conf_values['pyemma'][name] = value

# assign an alias to the wrapped module under 'config._impl'
sys.modules[__name__] = Wrapper(sys.modules[__name__])
