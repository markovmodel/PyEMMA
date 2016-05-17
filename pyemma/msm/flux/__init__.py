import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule
from pyemma.util.exceptions import PyEMMA_DeprecationWarning

warnings.warn("The pyemma.msm.flux module has been deprecated. "
              "You should import msmtools.flux now.", PyEMMA_DeprecationWarning)

sys.modules['pyemma.msm.flux'] = ShimModule(src='pyemma.msm.flux', mirror='msmtools.flux')

