import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule
from pyemma.util.exceptions import PyEMMA_DeprecationWarning

warnings.warn("The pyemma.msm.generation module has been deprecated. "
              "You should import msmtools.generation now.", PyEMMA_DeprecationWarning)

sys.modules['pyemma.msm.generation'] = ShimModule(src='pyemma.msm.generation', mirror='msmtools.generation')

