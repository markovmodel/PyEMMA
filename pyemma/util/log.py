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

from __future__ import absolute_import, print_function

import logging
from logging.config import dictConfig
import os.path
import warnings

import pkg_resources


__all__ = ['getLogger',
           ]

def_conf_file = pkg_resources.resource_filename('pyemma', 'logging.yml')
del pkg_resources

class LoggingConfigurationError(RuntimeError):
    pass

def setupLogging():
    """
    parses pyemma configuration file and creates a logger conf_values from that
    """
    from pyemma.util import config
    import yaml

    args = config.logging_config
    default = False

    if args.upper() == 'DEFAULT':
        default = True
        src = os.path.join(config.cfg_dir, 'logging.yml')
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
                with open(def_conf_file) as f:
                    D = yaml.load(f)
                    warnings.warn('Your set logging configuration could not '
                                  'be used. Used default as fallback.')
            except EnvironmentError as ee2:
                raise LoggingConfigurationError('Could not read either configured nor '
                                                'default logging configuration!\n%s' % ee)
        else:
            raise LoggingConfigurationError('could not handle default logging '
                                            'configuration file\n%s' % ee2)

    if D is None:
        raise LoggingConfigurationError('Empty logging config! Try using default config by'
                                        ' setting logging_conf=DEFAULT in pyemma.cfg')

    # this has not been set in PyEMMA version prior 2.0.2+
    D.setdefault('version', 1)
    # if the user has not explicitly disabled other loggers, we (contrary to Pythons
    # default value) do not want to override them.
    D.setdefault('disable_existing_loggers', False)

    # configure using the dict
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
            if f is not None and os.stat(f).st_size == 0:
                try:
                    os.remove(f)
                except OSError as o:
                    print("during removal of empty logfiles there was a problem: ", o)


def getLogger(name=None):
    # if name is not given, return a logger with name of the calling module.
    if not name:
        import traceback
        t = traceback.extract_stack(limit=2)
        name = t[0][0]

    return logging.getLogger(name)
