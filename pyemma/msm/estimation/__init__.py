import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule, ShimWarning

warnings.warn("The pyemma.msm.estimation module has been deprecated. "
              "You should import msmtools.estimation now.", ShimWarning)

sys.modules['pyemma.msm.estimation'] = ShimModule(src='pyemma.msm.estimation', mirror='msmtools.estimation')

from msmtools.estimation import *
