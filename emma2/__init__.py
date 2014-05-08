r"""
==========================================
    Emma2 - Emma's Markov Model Algorithms
==========================================
"""
import coordinates
import msm
import pmm
import util

# note that this works only for installed (or egg links created via setup.py develop)
import pkg_resources
try:
    __version__ = pkg_resources.get_distribution('emma2').version
except pkg_resources.DistributionNotFound:
    __version__ = 'undefined-cause-not-installed'
finally:
    del pkg_resources

