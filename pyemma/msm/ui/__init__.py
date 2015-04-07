r"""

=================================================
ui - MSM utility functions (:mod:`pyemma.msm.ui`)
=================================================

.. currentmodule:: pyemma.msm.ui

MSM-object
==========

.. autosummary::
   :toctree: generated/

   MSM - Markov state model object from a given transition matrix
   EstimatedMSM - Markov state model object estimated from data

ITS-object
==========

.. autosummary::
   :toctree: generated/

   ImpliedTimescales - ITS-object
   
"""
from .msm import *
from .timescales import *
from .chapman_kolmogorov import *
