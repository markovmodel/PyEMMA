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
   load_discrete_trajectory - read microstate trajectoryfrom binary file
   save_discrete_trajectory -  write microstate trajectory to binary file

Dense and sparse matrix io
==========================

.. autosummary::
   :toctree: generated/

   read_matrix - read matrix from ascii file
   write_matrix - write matrix to ascii file
   load_matrix - read matrix from binary file
   save_matrix - write matrix to binary file   

"""

from .api import *


