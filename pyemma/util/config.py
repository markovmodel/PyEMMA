# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import six
from six.moves import configparser

import os
import shutil
import sys
import warnings

from pyemma.util.files import mkdir_p
from pyemma.util.exceptions import ConfigDirectoryException

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

if six.PY2:
    class NotADirectoryError(Exception):
        pass


# indicate error during reading
class ReadConfigException(Exception):
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
    DEFAULT_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.pyemma')
    DEFAULT_CONFIG_FILE_NAME = 'pyemma.cfg'
    DEFAULT_LOGGING_FILE_NAME = 'logging.yml'

    __name__ = 'pyemma.util.config'
    __file__ = __file__

    def __init__(self, wrapped):
        # this is a SafeConfigParser instance
        self._conf_values = None

        # note we do not invoke the cfg_dir setter here, because we do not want anything to be created/copied yet.
        # first check if there is a config dir set via environment
        if 'PYEMMA_CFG_DIR' in os.environ:
            # TODO: probe?
            self._cfg_dir = os.environ['PYEMMA_CFG_DIR']
        # try to read default cfg dir
        elif os.path.isdir(self.DEFAULT_CONFIG_DIR) and os.access(self.DEFAULT_CONFIG_DIR, os.W_OK):
            self._cfg_dir = self.DEFAULT_CONFIG_DIR
        # use defaults, have no cfg_dir set.
        else:
            self._cfg_dir = ''
            from pyemma import __version__
            print("[PyEMMA {version}]: no configuration directory set or usable."
                  " Falling back to defaults.".format(version=__version__))

        try:
            self.load()
        except RuntimeError as re:
            warnings.warn("unable to read default configuration file. Logging and "
                          " progress bar handling could behave bad! Error: %s" % re)

        from pyemma.util.log import setup_logging, LoggingConfigurationError
        try:
            setup_logging(self)
        except LoggingConfigurationError as e:
            warnings.warn("Error during logging configuration. Logging might not be functional!"
                          "Error: %s" % e)

        # wrap this module
        self.wrapped = wrapped
        self.__wrapped__ = wrapped

    def __call__(self, ):
        return Wrapper(sys.modules[__name__])

    def load(self, filename=None):
        """ load runtime configuration from given filename.
        If filename is None try to read from default file from
        default location. """
        if not filename:
            filename = self.default_config_file

        files = self._cfgs_to_read()
        # insert last, so it will override all values,
        # which have already been set in previous files.
        files.insert(-1, filename)

        try:
            config = self.__read_cfg(files)
        except ReadConfigException as e:
            print('config.load("{file}") failed with {error}'.format(file=filename, error=e))
        else:
            self._conf_values = config

    def save(self, filename=None):
        """ Saves the runtime configuration to disk.

        Parameters
        ----------
        filename ; str or None, default=None
            writeable path to configuration filename. If None, use default location and filename.
        """
        if not filename:
            filename = self.DEFAULT_CONFIG_FILE_NAME
        else:
            filename = str(filename)
            # try to extract the path from filename and use is as cfg_dir
            head, tail = os.path.split(filename)
            if head:
                self._cfg_dir = head

            # we are search for .cfg files in cfg_dir so make sure it contains the proper extension.
            base, ext = os.path.splitext(tail)
            if ext != ".cfg":
                filename += ".cfg"

        filename = os.path.join(self.cfg_dir, filename)

        # if we have no cfg dir, try to create it first. Return if it failed.
        if not self.cfg_dir:
            try:
                self.cfg_dir = self.DEFAULT_CONFIG_DIR
            except ConfigDirectoryException as cde:
                print('Could not create configuration directory "{dir}"! config.save() failed.'
                      ' Please set a writeable location with config.cfg_dir = val. Error was {exc}'
                      .format(dir=self.cfg_dir, exc=cde))
                return

        try:
            with open(filename, 'w') as fh:
                self._conf_values.write(fh)
        except IOError as ioe:
            print("Save failed with error %s" % ioe)

    @property
    def used_filenames(self):
        """these filenames have been red to obtain basic configuration values."""
        return self._used_filenames

    @property
    def default_config_file(self):
        """ default config file living in PyEMMA package """
        return pkg_resources.resource_filename('pyemma', Wrapper.DEFAULT_CONFIG_FILE_NAME)

    @property
    def default_logging_file(self):
        return pkg_resources.resource_filename('pyemma', Wrapper.DEFAULT_LOGGING_FILE_NAME)

    def keys(self):
        return ['show_progress_bars',
                'use_trajectory_lengths_cache',
                'logging_config',
                ]

    @property
    def cfg_dir(self):
        """ PyEMMAs configuration directory (eg. ~/.pyemma)"""
        return self._cfg_dir

    @cfg_dir.setter
    def cfg_dir(self, pyemma_cfg_dir):
        """ Sets PyEMMAs configuration directory.
        Also creates it with some default files, if does not exists."""
        if not os.path.exists(pyemma_cfg_dir):
            try:
                mkdir_p(pyemma_cfg_dir)
            except EnvironmentError:
                raise ConfigDirectoryException("could not create configuration directory '%s'" % pyemma_cfg_dir)
            except NotADirectoryError:  # on Python 3
                raise ConfigDirectoryException("pyemma cfg dir (%s) is not a directory" % pyemma_cfg_dir)

        if not os.path.isdir(pyemma_cfg_dir):
            raise ConfigDirectoryException("%s is no valid directory" % pyemma_cfg_dir)
        if not os.access(pyemma_cfg_dir, os.W_OK):
            raise ConfigDirectoryException("%s is not writeable" % pyemma_cfg_dir)

        # give user the default cfg file, if its not there
        self.__copy_default_files_to_cfg_dir(pyemma_cfg_dir)
        self._cfg_dir = pyemma_cfg_dir

        print('*'*80, '\n',
              'Changed PyEMMAs config directory to "{dir}".\n'
              'To make this change permanent, export the environment variable'
              ' "PYEMMA_CFG_DIR" \nto point to this location. Eg. edit your .bashrc file!'.format(dir=pyemma_cfg_dir),
              '\n', '*' * 80, sep='')

    ### SETTINGS
    @property
    def logging_config(self):
        cfg = self._conf_values.get('pyemma', 'logging_config')
        if cfg == 'DEFAULT':
            cfg = os.path.join(self.cfg_dir, Wrapper.DEFAULT_LOGGING_FILE_NAME)
        return cfg

    @property
    def show_progress_bars(self):
        return self._conf_values.getboolean('pyemma', 'show_progress_bars')

    @show_progress_bars.setter
    def show_progress_bars(self, val):
        self._conf_values.set('pyemma', 'show_progress_bars', str(val))

    @property
    def use_trajectory_lengths_cache(self):
        return self._conf_values.getboolean('pyemma', 'use_trajectory_lengths_cache')

    @use_trajectory_lengths_cache.setter
    def use_trajectory_lengths_cache(self, val):
        self._conf_values.set('pyemma', 'use_trajectory_lengths_cache', str(val))

    def __copy_default_files_to_cfg_dir(self, target_dir):
        try:
            os.stat(self.default_config_file)
            os.stat(self.default_logging_file)
        except OSError:
            raise ConfigDirectoryException('Error during accessing default file "%s"' %
                                           self.default_config_file)
        files_to_copy = [
            self.default_config_file,
            self.default_logging_file,
        ]

        dests = [os.path.join(target_dir, os.path.basename(f)) for f in files_to_copy]
        for src, dest in zip(files_to_copy, dests):
            if not os.path.exists(dest):
                shutil.copyfile(src, dest)

    def __read_cfg(self, filenames):
        config = configparser.SafeConfigParser()

        try:
            self._used_filenames = config.read(filenames)
        except EnvironmentError as e:
            # note: this file is mission crucial, so fail badly if this is not readable.
            raise ReadConfigException("FATAL ERROR: could not read default configuration"
                                      " file %s\n%s" % (self.default_config_file, e))
        return config

    def _cfgs_to_read(self):
        """
        reads config files from various locations to build final config.
        """
        # use these files to extend/overwrite the conf_values.
        # Last red file always overwrites existing values!
        cfg = Wrapper.DEFAULT_CONFIG_FILE_NAME
        filenames = [
            self.default_config_file,
            cfg,  # conf_values in current directory
            os.path.join(os.path.expanduser('~' + os.path.sep), cfg),  # config in user dir
            '.pyemma.cfg',
        ]

        # look for user defined files
        if self.cfg_dir:
            from glob import glob
            filenames.extend(glob(self.cfg_dir + os.path.sep + "*.cfg"))
        return filenames

    # for dictionary like lookups
    def __getitem__(self, name):
        try:
            return self._conf_values.get('pyemma', name)
        except KeyError:  # re-try with default section
            return self._conf_values.get(name)

    def __setitem__(self, name, value):
        value = str(value)
        self._conf_values.set('pyemma', name, value)

# assign an alias to the wrapped module under 'config._impl'
sys.modules[__name__] = Wrapper(sys.modules[__name__])
