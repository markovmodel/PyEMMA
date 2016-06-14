
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

.. currentmodule:: pyemma.msm

User Functions
==============
For most users, the following high-level functions are sufficient to estimate msm models from data.
Expert users may want to construct Estimators or Models (see below) directly.

.. autosummary::
   :toctree: generated/

   markov_model
   timescales_msm
   its
   estimate_markov_model
   bayesian_markov_model
   tpt
   timescales_hmsm
   estimate_hidden_markov_model
   bayesian_hidden_markov_model

MSM classes
===========

**Estimators** to generate models from data. If you are not an expert user,
use the API functions above.

.. autosummary::
   :toctree: generated/

   ImpliedTimescales
   ChapmanKolmogorovValidator
   MaximumLikelihoodMSM
   BayesianMSM
   MaximumLikelihoodHMSM
   BayesianHMSM


**Models** of the kinetics or stationary properties of the data. 
If you are not an expert user, use the API functions above.

.. autosummary::
   :toctree: generated/

   MSM
   SampledMSM
   HMSM
   SampledHMSM
   ReactiveFlux
   PCCA


MSM functions (low-level API)
=============================
Low-level functions for estimation and analysis of transition matrices and io.

.. toctree::
   :maxdepth: 1

   msm.dtraj
   msm.generation
   msm.estimation
   msm.analysis
   msm.flux

"""
from __future__ import absolute_import as _

#####################################################
# Low-level MSM functions (imported from msmtools)
# backward compatibility to PyEMMA 1.2.x
# TODO: finally remove this stuff...
from pyemma.util._ext.shimmodule import ShimModule
analysis = ShimModule(src='pyemma.msm.analysis', mirror='msmtools.analysis')
estimation = ShimModule(src='pyemma.msm.estimation', mirror='msmtools.estimation')
generation = ShimModule(src='pyemma.msm.generation', mirror='msmtools.generation')
dtraj = ShimModule(src='pyemma.msm.dtraj', mirror='msmtools.dtraj')
io = dtraj
flux = ShimModule(src='pyemma.msm.flux', mirror='msmtools.flux')
del ShimModule
######################################################
from msmtools.analysis.dense.pcca import PCCA

#####################################################
# Estimators and models
from .estimators import MaximumLikelihoodMSM, BayesianMSM
from .estimators import MaximumLikelihoodHMSM, BayesianHMSM
from .estimators import ImpliedTimescales
from .estimators import ChapmanKolmogorovValidator

from .models import MSM, HMSM, SampledMSM, SampledHMSM, ReactiveFlux

# high-level api
from .api import *
