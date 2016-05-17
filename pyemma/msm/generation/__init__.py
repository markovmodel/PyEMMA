import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule, ShimWarning

warnings.warn("The pyemma.msm.generation module has been deprecated. "
              "You should import msmtools.generation now.", ShimWarning)

sys.modules['pyemma.msm.generation'] = ShimModule(src='pyemma.msm.generation', mirror='msmtools.generation')

from msmtools.generation import *
