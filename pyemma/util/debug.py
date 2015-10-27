''' registers some debug handlers on the importing Python process:
1. SIGUSR1 -> show a stack trace of the current frame
2. SIGUSR2 -> attach the Python debugger pdb to the current process



Created on 15.10.2015

@author: marscher
'''
from __future__ import absolute_import

import signal
from .log import getLogger

logger = None


def show_stacktrace(sig, frame):
    import traceback
    import StringIO

    global logger
    if logger is None:
        logger = getLogger('dbg')

    out = StringIO.StringIO()

    traceback.print_stack(frame, file=out)

    out.seek(0)
    trace = out.read()
    logger.info(trace)


def handle_pdb(sig, frame):
    import pdb

    pdb.Pdb().set_trace(frame)


def register_signal_handlers():
    # Register handler
    global logger

    signal.signal(42, show_stacktrace)
    signal.signal(43, handle_pdb)

def unregister_signal_handlers():
    pass
