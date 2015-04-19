r"""

.. currentmodule:: pyemma.msm

User API
========

.. autosummary::
   :toctree: generated/

   its
   markov_model
   estimate_markov_model
   tpt
   cktest

**MSM classes** encapsulating complex functionality. You don't need to construct these classes yourself, as this
is done by the user API functions above. Find here a documentation how to extract features from them.

.. autosummary::
   :toctree: generated/

   ui.MSM
   ui.EstimatedMSM
   ui.ImpliedTimescales
   flux.ReactiveFlux


MSM functions (low-level API)
=============================
Low-level functions for estimation and analysis of transition matrices and io.

.. toctree::
   :maxdepth: 1

   msm.io
   msm.generation
   msm.estimation
   msm.analysis
   msm.flux


"""

from . import analysis
from . import estimation
from . import generation
from . import io
from . import flux

from ui import ImpliedTimescales
from ui import MSM
from ui import EstimatedMSM
from flux import ReactiveFlux

from .api import *
