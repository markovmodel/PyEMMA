
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