# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from __future__ import absolute_import, print_function

import six
from six.moves.configparser import ConfigParser
import os
import shutil
import warnings

from pyemma.util.files import mkdir_p
from pyemma.util.exceptions import ConfigDirectoryException

import pkg_resources


# indicate error during reading
class ReadConfigException(Exception):
    pass

if six.PY2:
    class NotADirectoryError(Exception):
        pass

__all__ = ('Config', )


class Config(object):

    DEFAULT_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.pyemma')
    DEFAULT_CONFIG_FILE_NAME = 'pyemma.cfg'
    DEFAULT_LOGGING_FILE_NAME = 'logging.yml'

    def __init__(self):
        # this is a ConfigParser instance
        self._conf_values = None

        self._old_level = None

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
            print(Config._format_msg('config.load("{file}") failed with {error}'.format(file=filename, error=e)))
        else:
            self._conf_values = config

        # notice user?
        if self.show_config_notification and not self.cfg_dir:
            print(Config._format_msg("no configuration directory set or usable."
                                      " Falling back to defaults."))

    def __call__(self):
        # return a new instance, this is used for testing purposes only.
        return Config()

    def save(self, filename=None):
        """ Saves the runtime configuration to disk.

        Parameters
        ----------
        filename: str or None, default=None
            writeable path to configuration filename.
            If None, use default location and filename.
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

        # if we have no cfg dir, try to create it first. Return if it failed.
        if not self.cfg_dir or not os.path.isdir(self.cfg_dir) or not os.stat(self.cfg_dir) != os.W_OK:
            try:
                self.cfg_dir = self.DEFAULT_CONFIG_DIR
            except ConfigDirectoryException as cde:

                print(Config._format_msg('Could not create configuration directory "{dir}"! config.save() failed.'
                                          ' Please set a writeable location with config.cfg_dir = val. Error was {exc}'
                                          .format(dir=self.cfg_dir, exc=cde)))
                return

        filename = os.path.join(self.cfg_dir, filename)

        try:
            with open(filename, 'w') as fh:
                self._conf_values.write(fh)
        except IOError as ioe:
            print(Config._format_msg("Save failed with error %s" % ioe))

    @property
    def used_filenames(self):
        """these filenames have been red to obtain basic configuration values."""
        return self._used_filenames

    @property
    def default_config_file(self):
        """ default config file living in PyEMMA package """
        return pkg_resources.resource_filename('pyemma', Config.DEFAULT_CONFIG_FILE_NAME)

    @property
    def default_logging_file(self):
        """ default logging configuration"""
        return pkg_resources.resource_filename('pyemma', Config.DEFAULT_LOGGING_FILE_NAME)

    def keys(self):
        """ valid configuration keys"""
        return self._conf_values.options('pyemma')

    @property
    def cfg_dir(self):
        """ PyEMMAs configuration directory (eg. ~/.pyemma)"""
        return self._cfg_dir

    @cfg_dir.setter
    def cfg_dir(self, pyemma_cfg_dir):
        """ Sets PyEMMAs configuration directory.
        Also creates it with some default files, if does not exists. """
        if not os.path.exists(pyemma_cfg_dir):
            try:
                mkdir_p(pyemma_cfg_dir)
            except NotADirectoryError:  # on Python 3
                raise ConfigDirectoryException("pyemma cfg dir (%s) is not a directory" % pyemma_cfg_dir)
            except EnvironmentError:
                raise ConfigDirectoryException("could not create configuration directory '%s'" % pyemma_cfg_dir)

        if not os.path.isdir(pyemma_cfg_dir):
            raise ConfigDirectoryException("%s is no valid directory" % pyemma_cfg_dir)
        if not os.access(pyemma_cfg_dir, os.W_OK):
            raise ConfigDirectoryException("%s is not writeable" % pyemma_cfg_dir)

        # give user the default cfg file, if its not there
        self.__copy_default_files_to_cfg_dir(pyemma_cfg_dir)
        self._cfg_dir = pyemma_cfg_dir

        if self.show_config_notification:
            stars = '*' * 80
            print(stars, '\n',
                  'Changed PyEMMAs config directory to "{dir}".\n'
                  'To make this change permanent, export the environment variable'
                  ' "PYEMMA_CFG_DIR" \nto point to this location. Eg. edit your .bashrc file!'
                  .format(dir=pyemma_cfg_dir), '\n', stars, sep='')

    ### SETTINGS
    @property
    def logging_config(self):
        """ currently used logging configuration file. Can not be changed during runtime. """
        cfg = self._conf_values.get('pyemma', 'logging_config')
        if cfg == 'DEFAULT':
            cfg = os.path.join(self.cfg_dir, Config.DEFAULT_LOGGING_FILE_NAME)
        return cfg

    # FIXME: how should we re-initialize logging without interfering with existing loggers?
    #@logging_config.setter
    #def logging_config(self, config):
    #    """ Try to re-initialize logging system for package 'pyemma'.
    #
    #    Parameters
    #    ----------
    #    config: dict
    #        A dictionary which contains at least the keys 'loggers' and 'handlers'.
    #    """
    #    from pyemma.util.log import setup_logging
    #   #config['incremental'] = True
    #    setup_logging(self, config)

    @property
    def mute(self):
        """ Switch this to True, to tell PyEMMA not to use progress bars and logging to console. """
        return self._conf_values.getboolean('pyemma', 'mute')

    @mute.setter
    def mute(self, value):
        value = bool(value)
        import logging
        if value:
            self.show_progress_bars = False
            self._old_level = logging.getLogger('pyemma').level
            logging.getLogger('pyemma').setLevel('CRITICAL')
        else:
            self.show_progress_bars = True

            if self._old_level is not None:
                logging.getLogger('pyemma').setLevel(self._old_level)
                self._old_level = None

        self._conf_values.set('pyemma', 'mute', str(value))

    @property
    def traj_info_max_entries(self):
        """ How many entries (files) the trajectory info cache can hold.
        The cache will forget the least recently used entries when this limit is hit."""
        return self._conf_values.getint('pyemma', 'traj_info_max_entries')

    @traj_info_max_entries.setter
    def traj_info_max_entries(self, val):
        self._conf_values.set('pyemma', 'traj_info_max_entries', str(val))

    @property
    def traj_info_max_size(self):
        """ Maximum trajectory info cache size in bytes.
        The cache will forget the least recently used entries when this limit is hit."""
        return self._conf_values.getint('pyemma', 'traj_info_max_size')

    @traj_info_max_size.setter
    def traj_info_max_size(self, val):
        val = str(int(val))
        self._conf_values.set('pyemma', 'traj_info_max_size', val)

    @property
    def show_progress_bars(self):
        """Show progress bars for heavy computations?"""
        return self._conf_values.getboolean('pyemma', 'show_progress_bars')

    @show_progress_bars.setter
    def show_progress_bars(self, val):
        self._conf_values.set('pyemma', 'show_progress_bars', str(val))

    @property
    def use_trajectory_lengths_cache(self):
        """ Shall the trajectory info cache be used to remember attributes of trajectory files.

        It is strongly recommended to use the cache especially for XTC files, because this will speed up
        reader creation a lot.
        """
        return self._conf_values.getboolean('pyemma', 'use_trajectory_lengths_cache')

    @use_trajectory_lengths_cache.setter
    def use_trajectory_lengths_cache(self, val):
        self._conf_values.set('pyemma', 'use_trajectory_lengths_cache', str(val))

    @property
    def show_config_notification(self):
        """ """
        return self._conf_values.getboolean('pyemma', 'show_config_notification')

    @show_config_notification.setter
    def show_config_notification(self, val):
        self._conf_values.set('pyemma', 'show_config_notification', str(val))

    @property
    def coordinates_check_output(self):
        """ Enabling this option will check for invalid output (NaN, Inf) in pyemma.coordinates """
        return self._conf_values.getboolean('pyemma', 'coordinates_check_output')

    @coordinates_check_output.setter
    def coordinates_check_output(self, val):
        self._conf_values.set('pyemma', 'coordinates_check_output', str(val))

    @property
    def check_version(self):
        """ Check for the latest release online.

        Disable this if you have privacy concerns.
        We currently collect:

         * Python version
         * PyEMMA version
         * operating system
         * MAC address

        See :doc:`legal` for further information.
        """
        return self._conf_values.getboolean('pyemma', 'check_version')

    @check_version.setter
    def check_version(self, val):
        self._conf_values.set('pyemma', 'check_version', str(val))

    @property
    def default_chunksize(self):
        """ default chunksize to use for coordinate transformations, only intergers with suffix [k,m,g]"""
        return self._conf_values.get('pyemma', 'default_chunksize')

    @default_chunksize.setter
    def default_chunksize(self, val):
        from pyemma.util.units import string_to_bytes
        # check for parsing exceptions
        string_to_bytes(val)
        self._conf_values.set('pyemma', 'default_chunksize', str(val))

    ### FIlE HANDLING

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
        config = ConfigParser()

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
        cfg = Config.DEFAULT_CONFIG_FILE_NAME
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

    def __setattr__(self, key, value):
        if key.startswith('_') or key == 'cfg_dir':
            pass
        elif key not in self.keys():
            raise ValueError('Not a valid configuration key: "%s"' % key)
        super(Config, self).__setattr__(key, value)

    @staticmethod
    def _format_msg(msg):
        from pyemma import __version__
        return "[PyEMMA {version}] {msg}".format(version=__version__, msg=msg)

    def __repr__(self):
        cfg = {key: self[key] for key in self.keys()}
        return "[PyEMMA config. State = {cfg}]".format(cfg=cfg)

