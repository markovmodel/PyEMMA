r"""

=============================================================
analysis - MSM analysis functions (:mod:`emma2.msm.analysis`)
=============================================================

.. currentmodule:: emma2.msm.analysis
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin.trendelkampschroer@gmail.com>

Validation
==========

.. autosummary::
   :toctree: generated/

   is_transition_matrix - Positive entries and rows sum to one
   is_rate_matrix - Nonpositive off-diagonal entries and rows sum to zero
   is_ergodic - Irreducible matrix
   is_reversible - Symmetric with respect to some probability vector pi

Decomposition
=============

Decomposition routines use the scipy LAPACK bindings for dense
numpy-arrays and the ARPACK bindings for scipy sparse matrices.

.. autosummary::
   :toctree: generated/

   stationary_distribution - Invariant vector from eigendecomposition 
   eigenvalues - Spectrum via eigenvalue decomposition
   eigenvectors - Right or left eigenvectors
   rdl_decomposition - Full decomposition into eigenvalues and eigenvectors
   timescales - Implied timescales from eigenvalues

Expectations
============

.. autosummary::
   :toctree: generated/

   expectation - Equilibrium expectation value of an observable
   expected_counts - Count matrix expected for given initial distribution
   expected_counts_stationary - Count matrix expected for equilibrium distribution

Passage times, committors, TPT, PCCA
=====================================

.. autosummary::
   :toctree: generated/

   mfpt - Mean first-passage time
   committor - Forward and backward commitor
   tpt - Transition paths and fluxes
   pcca - Perron cluster center analysis

Fingerprints
============

.. autosummary::
   :toctree: generated/

   fingerprint_correlation
   fingerprint_relaxation
   evaluate_fingerprint
   correlation
   relaxation

Sensitivity analysis
====================

.. autosummary::
   :toctree: generated/

   stationary_distribution_sensitivity
   eigenvalue_sensitivity
   timescale_sensitivity
   eigenvector_sensitivity
   mfpt_sensitivity
   committor_sensitivity

"""

from __future__ import division, print_function, absolute_import

from .api import *

__all__=[s for s in dir() if not s.startswith('_')]
