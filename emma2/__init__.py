"""
    Emma2 - Emma's Markov Model Algorithms 
"""

import coordinates
import msm
import pmm
import util

# scipy.sparse.diags function introduced in scipy 0.11, provide a fallback for
# users of older versions
import scipy.sparse
try:
    scipy.sparse.__dict__.get('diags')
except KeyError:
    scipy.sparse.__dict__['diags'] = util.numeric.diags
