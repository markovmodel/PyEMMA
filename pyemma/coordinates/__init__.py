
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
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
.. currentmodule:: pyemma.coordinates

User API
========

**Data handling and IO**

.. autosummary::
   :toctree: generated/

   featurizer
   load
   source
   pipeline
   discretizer
   save_traj
   save_trajs

**Transformations**

.. autosummary::
   :toctree: generated/

   pca
   tica

**Clustering Algorithms**

.. autosummary::
   :toctree: generated/

   cluster_kmeans
   cluster_regspace
   cluster_uniform_time
   assign_to_centers

Classes
=======
**Coordinate classes** encapsulating complex functionality. You don't need to
construct these classes yourself, as this is done by the user API functions above.
Find here a documentation how to extract features from them.

.. autosummary::
   :toctree: generated/

   pipelines.Pipeline
   transform.PCA
   transform.TICA

"""
from .api import *