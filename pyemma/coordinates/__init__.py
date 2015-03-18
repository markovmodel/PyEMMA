r"""
=========================================================
coordinates (:mod:`pyemma.coordinates`)
=========================================================

.. currentmodule:: pyemma.coordinates


User-API
========

The class which links input (readers), transformers (PCA, TICA) and clustering
together is the :func:`discretizer`. It builds up a pipeline to process your data
into discrete state space.

..autosummary::
  :toctree: generated/

Readers
-------
.. autosummary::
   :toctree: generated/

   feature_reader
   memory_reader

Transformations
---------------
.. autosummary::
   :toctree: generated/

   pca
   tica

Clustering Algorithms
---------------------
.. autosummary::
   :toctree: generated/

   cluster_kmeans
   cluster_regspace
   cluster_uniform_time
   cluster_assign_centers

"""
from .api import *
