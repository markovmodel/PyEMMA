
r"""

============================================
plots - Plotting tools (:mod:`pyemma.plots`)
============================================

.. currentmodule:: pyemma.plots

User-API
========

.. autosummary::
   :toctree: generated/

   plot_implied_timescales
   scatter_contour
   plot_markov_model
   plot_flux
   NetworkPlot

"""
from __future__ import absolute_import
from .timescales import plot_implied_timescales
from .plots2d import contour, scatter_contour, plot_free_energy
from .networks import plot_markov_model, plot_flux, plot_network, NetworkPlot
from .markovtests import plot_cktest
