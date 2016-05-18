import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule
from pyemma.util.exceptions import PyEMMA_DeprecationWarning

warnings.warn("The pyemma.msm.estimation module has been deprecated. "
              "You should import msmtools.estimation now.", PyEMMA_DeprecationWarning)

sys.modules['pyemma.msm.estimation'] = ShimModule(src='pyemma.msm.estimation', mirror='msmtools.estimation')

