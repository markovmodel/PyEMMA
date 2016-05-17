import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule, ShimWarning

warnings.warn("The pyemma.msm.analysis module has been deprecated. "
              "You should import msmtools.analysis now.", ShimWarning)

sys.modules['pyemma.msm.analysis'] = ShimModule(src='pyemma.msm.analysis', mirror='msmtools.analysis')

from msmtools.analysis import *
