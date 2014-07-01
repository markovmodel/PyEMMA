r"""

===================================================================
estimation - MSM estimation from data (:mod:`emma2.msm.estimation`)
===================================================================

.. currentmodule:: emma2.msm.estimation
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin.trendelkampschroer@gmail.com>

Countmatrix
===========

.. autosummary::
   :toctree: generated/

   count_matrix - estimate count matrix from discrete trajectories

Connectivity
============

.. autosummary::
   :toctree: generated/

   connected_sets - Find connected subsets
   largest_connected_set - Find largest connected set
   largest_connected_submatrix - Count matrix on largest connected set
   is_connected - Test count matrix connectivity

Estimation
==========

.. autosummary::
   :toctree: generated/

   transition_matrix - Estimate transition matrix
   log_likelihood

Sampling
========

.. autosummary::
   :toctree: generated/

   tmatrix_sampler - Random sample from transition matrix posterior

"""

from .api import * 
