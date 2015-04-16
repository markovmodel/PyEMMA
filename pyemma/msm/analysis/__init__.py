
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""

==============================================================
analysis - MSM analysis functions (:mod:`pyemma.msm.analysis`)
==============================================================

.. currentmodule:: pyemma.msm.analysis

This module contains functions to analyze a created Markov model, which is
specified with a transition matrix T.

Validation
==========

.. autosummary::
   :toctree: generated/

   is_transition_matrix - Positive entries and rows sum to one
   is_tmatrix
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
   statdist
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

Committors and PCCA
===================

.. autosummary::
   :toctree: generated/

   committor - Forward and backward committor
   pcca - Perron cluster center analysis

Fingerprints
============

.. autosummary::
   :toctree: generated/

   fingerprint_correlation
   fingerprint_relaxation
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