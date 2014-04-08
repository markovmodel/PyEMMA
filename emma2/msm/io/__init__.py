r"""

=================================================================
io - Input and output for MSM related files (:mod:`emma2.msm.io`)
=================================================================

.. currentmodule:: emma2.msm.io

Discrete trajectory io
======================

.. autosummary::
   :toctree: generated/
   
   read_discrete_trajectory - read microstate trajectoryfrom ascii file
   write_discrete_trajectory - write microstate trajectory to ascii file

"""
from __future__ import division, print_function, absolute_import

from .api import *

__all__=[s for s in dir() if not s.startswith('_') and
         s not in ['division', 'print_function', 'absolute_import']]
