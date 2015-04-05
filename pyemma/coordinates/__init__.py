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
   input
   discretizer
   save_traj
   save_trajs

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
