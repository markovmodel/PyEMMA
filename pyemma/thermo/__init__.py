# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
.. currentmodule:: pyemma.thermo

User-Functions
==============
For most users, the following high-level functions are sufficient to
estimate models from data.

.. autosummary::
   :toctree: generated/thermo-api

   estimate_umbrella_sampling
   estimate_multi_temperature
   dtram
   wham
   tram

Thermo classes
==============
**Estimators** to generate models from data. If you are not an expert user,
use the API functions above.

.. autosummary::
    :toctree: generated/thermo-estimators

    StationaryModel
    MEMM
    WHAM
    DTRAM
    TRAM

"""

from pyemma.thermo.models import StationaryModel, MEMM
from pyemma.thermo.estimators import WHAM, DTRAM, TRAM, EmptyState

# high-level api
from .api import *
