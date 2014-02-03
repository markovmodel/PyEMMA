"""
    Emma2 - Emma's Markov Model Algorithms 
"""

import coordinates
import msm
import pmm
import util

""" global logging support. See documentation of python library
logging module for more information"""
from util.log import log


# scipy.sparse.diags function introduced in scipy 0.11, provide a fallback for
# users of older versions
import scipy.sparse
try:
    scipy.sparse.__dict__.get('diags')
except KeyError:
    scipy.sparse.__dict__['diags'] = util.numeric.diags
