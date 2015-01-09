r"""

===============================================================================
transform - Transformation Utilities (:mod:`pyemma.coordinates.transform`)
===============================================================================

.. currentmodule:: pyemma.coordinates.transform

Order parameters
================

.. autosummary::
  :toctree: generated/

  createtransform_selection - select a given set of atoms
  createtransform_distances - compute intramolecular distances
  createtransform_angles - compute angles
  createtransform_dihedrals - compute dihedral
  createtransform_minrmsd - compute minrmsd distance


Complex transformations
========================

.. autosummary::
  :toctree: generated/

  pca - principal components
  tica - time independent components
  transform_file
  transform_trajectory

"""
from .api import *
from .util import *
