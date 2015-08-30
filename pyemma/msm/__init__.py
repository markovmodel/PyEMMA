
r"""

.. currentmodule:: pyemma.msm

User API
========
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


**Estimators** to generate models from data. If you are not an expert user,
use the API functions above.

.. autosummary::
   :toctree: generated/

   ImpliedTimescales
   MaximumLikelihoodMSM
   BayesianMSM
   MaximumLikelihoodHMSM
   BayesianHMSM


**Models** of the kinetics or stationary properties of the data. 
If you are not an expert user, use the API functions above.

.. autosummary::
   :toctree: generated/

   MSM
   EstimatedMSM
   SampledMSM
   HMSM
   EstimatedHMSM
   SampledHMSM
   ReactiveFlux


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
from __future__ import absolute_import, print_function

#####################################################
# Low-level MSM functions (imported from msmtools)
# backward compatibility to PyEMMA 1.2.x
from msmtools import analysis, estimation, generation, dtraj, flux
from msmtools.flux import ReactiveFlux
io = dtraj

#####################################################
# Estimators and models
from .estimators import MaximumLikelihoodMSM, BayesianMSM
from .estimators import MaximumLikelihoodHMSM, BayesianHMSM
from .estimators import ImpliedTimescales
from .estimators import EstimatedMSM, EstimatedHMSM

from .models import MSM, HMSM, SampledMSM, SampledHMSM

# high-level api
from .api import *
