
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2018 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


r"""

============================================
plots - Plotting tools (:mod:`pyemma.plots`)
============================================

.. currentmodule:: pyemma.plots

User-API
========

**Graph plots**

.. autosummary::
   :toctree: generated/

   plot_implied_timescales
   plot_cktest

**Contour plots**

.. autosummary::
   :toctree: generated/

   plot_density
   plot_free_energy
   plot_contour
   plot_state_map
   scatter_contour

**Network plots**

.. autosummary::
   :toctree: generated/

   plot_markov_model
   plot_flux
   plot_network

Classes
========

.. autosummary::
   :toctree: generated/

   NetworkPlot

"""
from .timescales import plot_implied_timescales
from .plots2d import contour, scatter_contour
from .plots2d import plot_density
from .plots2d import plot_free_energy
from .plots2d import plot_contour
from .plots2d import plot_state_map
from .networks import plot_markov_model, plot_flux, plot_network, NetworkPlot
from .markovtests import plot_cktest
from .thermoplots import *
from .plots1d import plot_feature_histograms
