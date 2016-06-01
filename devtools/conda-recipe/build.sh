#!/bin/bash
$PYTHON setup.py install
stmnt="from __future__ import print_statement; import pyemma; print(pyemma.version)"
export PYEMMA_SETUP_VERSION $($PYTHON -c $stmnt)

