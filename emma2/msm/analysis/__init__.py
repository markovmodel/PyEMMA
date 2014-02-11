r"""

=============================================================
analysis - MSM analysis functions (:mod:`emma2.msm.analysis`)
=============================================================

.. currentmodule:: emma2.msm.analysis

Transition matrix validation
============================

.. autosummary::
   :toctree: generated/

   is_transition_matrix - positive entries and rows sum to one
   is_rate_matrix - nonpositive off-diagonal entries and rows sum to zero
   is_ergodic - irreducible matrix
   is_reversible - symmetric with respect to some probability vector pi

Transition matrix decomposition
===============================

Decomposition routines use the scipy LAPACK bindings for dense
numpy-arrays and the ARPACK bindings for scipy sparse matrices.

.. autosummary::
   :toctree: generated/

   stationary_distribution - invariant vector from eigendecomposition 
   eigenvalues - spectrum via eigenvalue decomposition
   eigenvectors - right or left eigenvectors
   rdl_decomposition - full decomposition into eigenvalues and eigenvectors
   timescales - implied timescales from eigenvalues

"""

from __future__ import division, print_function, absolute_import

from .api import *

__all__=[s for s in dir() if not s.startswith('_')]
