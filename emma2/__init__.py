"""
    Emma2 package
"""

import pkg_resources
pkg_resources.require('numpy >= 1.8.0')

import coordinates
import msm
import pmm
import util

""" global logging support. See documentation of python library
logging module for more information"""
from util.log import log
