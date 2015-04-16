
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

====================================================================
estimation - MSM estimation from data (:mod:`pyemma.msm.estimation`)
====================================================================

.. currentmodule:: pyemma.msm.estimation

Countmatrix
===========

.. autosummary::
   :toctree: generated/

   count_matrix - estimate count matrix from discrete trajectories
   cmatrix - estimate count matrix from discrete trajectories

Connectivity
============

.. autosummary::
   :toctree: generated/

   connected_sets - Find connected subsets
   largest_connected_set - Find largest connected set
   largest_connected_submatrix - Count matrix on largest connected set
   connected_cmatrix
   is_connected - Test count matrix connectivity

Estimation
==========

.. autosummary::
   :toctree: generated/

   transition_matrix - Estimate transition matrix
   tmatrix
   log_likelihood
   tmatrix_cov
   error_perturbation


Sampling
========

.. autosummary::
   :toctree: generated/

   tmatrix_sampler - Random sample from transition matrix posterior

Bootstrap
=========

.. autosummary::
   :toctree: generated/

   bootstrap_counts
   bootstrap_trajectories

Priors
======

.. autosummary::
   :toctree: generated/
   
   prior_neighbor
   prior_const
   prior_rev


"""
from .api import *