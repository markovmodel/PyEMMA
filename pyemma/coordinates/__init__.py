r"""
=========================================================
coordinates (:mod:`pyemma.coordinates`)
=========================================================

.. currentmodule:: pyemma.coordinates


User-API
========

The class which links input (readers), transformers (PCA, TICA) and clustering
together is the **Discretizer**. It builds up a pipeline to process your data
into discrete state space. The API function **discretizer** creates it.


Data handling and IO
--------------------
.. autosummary::
   :toctree: generated/

   featurizer
   load
   source
   pipeline
   discretizer
   save_traj
   save_trajs

Deprecated Readers
--------------------
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
   assign_to_centers


"""
from .api import *
