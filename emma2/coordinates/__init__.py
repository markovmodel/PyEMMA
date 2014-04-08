r"""
===================================================================
coordinates - Coordinate transformations (:mod:`emma2.coordinates`)
===================================================================

.. currentmodule:: emma2.coordinates


Submodules
----------

.. autosummary::
   :toctree: generated/

   clustering - clustering of coordinates
   io - in and output
   transform - perform coordinate transformations
   tica - perform PCA and TICA on data
"""
from __future__ import division, print_function, absolute_import

import emma2.coordinates.tica
import emma2.coordinates.transform

__all__=[s for s in dir() if not s.startswith('_')]
