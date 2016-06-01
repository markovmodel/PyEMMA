#!/bin/bash
$PYTHON setup.py install
version=$($PYTHON -c "from __future__ import print_function; import pyemma; print(pyemma.__version__)")
export PYEMMA_VERSION=$version 
