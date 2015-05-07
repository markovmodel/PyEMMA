
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

.. currentmodule:: pyemma.msm

User API
========

.. autosummary::
   :toctree: generated/

   its
   markov_model
   estimate_markov_model
   tpt
   cktest

**MSM classes** encapsulating complex functionality. You don't need to construct these classes yourself, as this
is done by the user API functions above. Find here a documentation how to extract features from them.

.. autosummary::
   :toctree: generated/

   ui.MSM
   ui.EstimatedMSM
   ui.ImpliedTimescales
   flux.ReactiveFlux


MSM functions (low-level API)
=============================
Low-level functions for estimation and analysis of transition matrices and io.

.. toctree::
   :maxdepth: 1

   msm.io
   msm.generation
   msm.estimation
   msm.analysis
   msm.flux


"""

from . import analysis
from . import estimation
from . import generation
from . import io
from . import flux

from ui import ImpliedTimescales
from ui import MSM
from ui import EstimatedMSM
from ui import HMSM
from ui import EstimatedHMSM
from flux import ReactiveFlux

from .api import *