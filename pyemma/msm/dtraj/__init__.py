import sys
import warnings

from pyemma.util._ext.shimmodule import ShimModule
from pyemma.util.exceptions import PyEMMA_DeprecationWarning

warnings.warn("The pyemma.msm.dtraj module has been deprecated. "
              "You should import msmtools.dtraj now.", PyEMMA_DeprecationWarning)

sys.modules['pyemma.msm.dtraj'] = ShimModule(src='pyemma.msm.dtraj', mirror='msmtools.dtraj')
#sys.modules['pyemma.msm.io'] = sys.modules['pyemma.msm.dtraj']

