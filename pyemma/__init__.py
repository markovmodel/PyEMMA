r"""
=======================================
PyEMMA - Emma's Markov Model Algorithms
=======================================
"""
from . import coordinates
from . import msm
from . import util
from . import plots

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
