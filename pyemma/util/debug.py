
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

''' This module registers some debug handlers upon the call of
:py:func:register_signal_handlers

1. Show a stack trace of the current frame
2. Attach the Python debugger PDB to the current process

Created on 15.10.2015

@author: marscher
'''
from __future__ import absolute_import, print_function

import signal
from logging import getLogger

_logger = None

SIGNAL_STACKTRACE = 42
SIGNAL_PDB = 43


def _show_stacktrace(sig, frame):
    import traceback
    from six import StringIO

    global _logger
    if _logger is None:
        _logger = getLogger('pyemma.dbg')

    out = StringIO()

    traceback.print_stack(frame, file=out)

    out.seek(0)
    trace = out.read()
    _logger.info(trace)

def _handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


def register_signal_handlers():
    r""" Registeres some debug helping functions at the current Python process.

    Namely a stack trace generator (which uses the logging system) and a debugger
    attaching function.
    To trigger them, send the corresponding signal to the process::

    1. kill -42 $PID # obtain a stack trace
    2. kill -43 $PID # attach the debugger to the current stack frame

    For the first signal, a stack trace originating from the current frame will
    be logged using PyEMMAs logging system.

    The second signal stops your program at the current stack frame by using
    Pythons internal debugger PDB.
    See https://docs.python.org/2/library/pdb.html for information on how to use
    the debugger.

    To obtain a stack trace, just send the signal 42 to the current Python process id.
    This id can be obtained via:

    >>> import os # doctest: +SKIP
    >>> os.getpid() # doctest: +SKIP
    34588

    To send the signal you can use kill on Linux and OSX::

    kill -42 34588
    """
    signal.signal(SIGNAL_STACKTRACE, _show_stacktrace)
    signal.signal(SIGNAL_PDB, _handle_pdb)


def unregister_signal_handlers():
    """ set signal handlers to default """
    signal.signal(SIGNAL_STACKTRACE, signal.SIG_IGN)
    signal.signal(SIGNAL_PDB, signal.SIG_IGN)
