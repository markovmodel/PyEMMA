r"""

=============================================================
analysis - MSM analysis functions (:mod:`emma2.msm.analysis`)
=============================================================

.. currentmodule:: emma2.msm.analysis

Assessment of MSM properties
============================

.. autosummary::
   :toctree: generated/

   is_transition_matrix - check for stochasticity

"""

from __future__ import division, print_function, absolute_import

from .api import *

__all__=[s for s in dir() if not s.startswith('_')]
