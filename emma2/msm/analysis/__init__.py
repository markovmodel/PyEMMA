r"""

=============================================================
analysis - MSM analysis functions (:mod:`emma2.msm.analysis`)
=============================================================

.. currentmodule:: emma2.msm.analysis

This module contains functions to analyze a created Markov model, which is
specified with a transition matrix T.

Validation
==========

.. autosummary::
   :toctree: generated/

   is_transition_matrix - Positive entries and rows sum to one
   is_rate_matrix - Nonpositive off-diagonal entries and rows sum to zero
   is_connected - Irreducible matrix
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

Expected counts
=================

.. autosummary::
   :toctree: generated/

   expected_counts - Count matrix expected for given initial distribution
   expected_counts_stationary - Count matrix expected for equilibrium distribution

Passage times
=============

.. autosummary::
   :toctree: generated/

   mfpt - Mean first-passage time

Committors, TPT
===============

.. autosummary::
   :toctree: generated/

   committor - Forward and backward commitor
   tpt_flux - Flux network for reaction
   tpt_netflux - Netflux network
   tpt_totalflux - Total flux for reaction
   tpt_rate - Reaction rate
   tpt - Transition path theory object

PCCA
====

.. autosummary::
   :toctree: generated/

   pcca - Perron cluster center analysis

Fingerprints
============

.. autosummary::
   :toctree: generated/

   fingerprint_correlation
   fingerprint_relaxation
   evaluate_fingerprint
   expectation - Equilibrium expectation value of an observable
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
   expectation_sensitivity

"""
from .api import *
