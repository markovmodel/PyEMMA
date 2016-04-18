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
'''
Created on 15.10.2013

@author: marscher
'''

from __future__ import absolute_import

import logging
from logging.config import dictConfig
import os.path
import warnings

from pyemma.util.annotators import deprecated

__all__ = ['getLogger',
           ]


class LoggingConfigurationError(RuntimeError):
    pass


def setup_logging(config, D=None):
    """ set up the logging system with the configured (in pyemma.cfg) logging config (logging.yml)
    @param config: instance of pyemma.config module (wrapper)
    """
    if not D:
        import yaml

        args = config.logging_config
        default = False

        if args.upper() == 'DEFAULT':
            default = True
            src = config.default_logging_file
        else:
            src = args

        # first try to read configured file
        try:
            with open(src) as f:
                D = yaml.load(f)
        except EnvironmentError as ee:
            # fall back to default
            if not default:
                try:
                    with open(config.default_logging_file) as f2:
                        D = yaml.load(f2)
                except EnvironmentError as ee2:
                    raise LoggingConfigurationError('Could not read either configured nor '
                                                    'default logging configuration!\n%s' % ee2)
            else:
                raise LoggingConfigurationError('could not handle default logging '
                                                'configuration file\n%s' % ee)

        if D is None:
            raise LoggingConfigurationError('Empty logging config! Try using default config by'
                                            ' setting logging_conf=DEFAULT in pyemma.cfg')
    assert D

    # this has not been set in PyEMMA version prior 2.0.2+
    D.setdefault('version', 1)
    # if the user has not explicitly disabled other loggers, we (contrary to Pythons
    # default value) do not want to override them.
    D.setdefault('disable_existing_loggers', False)

    # configure using the dict
    try:
        dictConfig(D)
    except ValueError as ve:
        # issue with file handler?
        if 'files' in str(ve) and 'rotating_files' in D['handlers']:
            print("cfg dir", config.cfg_dir)
            new_file = os.path.join(config.cfg_dir, 'pyemma.log')
            warnings.warn("set logfile to %s, because there was"
                          " an error writing to the desired one" % new_file)
            D['handlers']['rotating_files']['filename'] = new_file
        else:
            raise
        dictConfig(D)

    # get log file name of pyemmas root logger
    logger = logging.getLogger('pyemma')
    log_files = [getattr(h, 'baseFilename', None) for h in logger.handlers]

    import atexit
    @atexit.register
    def clean_empty_log_files():
        # gracefully shutdown logging system
        logging.shutdown()
        for f in log_files:
            if f is not None and os.path.exists(f):
                try:
                    if os.stat(f).st_size == 0:
                        os.remove(f)
                except OSError as o:
                    print("during removal of empty logfiles there was a problem: ", o)

@deprecated("use logging.getLogger")
def getLogger(name=None):
    # if name is not given, return a logger with name of the calling module.
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        name = t[0][0]

    return logging.getLogger(name)
