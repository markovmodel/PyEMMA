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

Priors
======

.. autosummary::
   :toctree: generated/
   
   prior_neighbor
   prior_const
   prior_rev


"""
from .api import *
