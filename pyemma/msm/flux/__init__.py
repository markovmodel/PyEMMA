r"""

====================================================================
flux - Reactive flux an transition pathways (:mod:`pyemma.msm.flux`)
====================================================================

.. currentmodule:: pyemma.msm.flux

This module contains functions to compute reactive flux networks and
find dominant reaction pathways in such networks.

TPT-object
==========

.. autosummary::
   :toctree: generated/

   tpt
   ReactiveFlux

Reactive flux
=============

.. autosummary::
   :toctree: generated/

   flux_matrix - TPT flux network
   to_netflux - Netflux from gross flux
   flux_production - Net flux-production for all states
   flux_producers
   flux_consumers
   coarsegrain

Reaction rates and fluxes
=========================

.. autosummary::
   :toctree: generated/

   total_flux
   rate
   mfpt
   

Pathway decomposition
=====================

.. autosummary::
   :toctree: generated/

   pathways

"""
from .api import *
from .reactive_flux import *

