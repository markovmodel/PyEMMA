.. _ref_api:

Coordinates
===========
The *coordinates* package contains tools to select features from MD-trajectories
to assign them to a discrete state space, which will be later used in Markov
modeling.

.. toctree::
   :maxdepth: 1
   
   coordinates.io
   coordinates.transform
   coordinates.clustering

Markov State Models
===================
The *msm* package provides functions to estimate, analyze and generate
Markov state models. All public functions accept dense NumPy and SciPy
sparse types and distinguish them automatically, so the optimal
underlying solution for a problem is choosen.


.. toctree::
   :maxdepth: 1
   
   msm
   msm.io
   msm.estimation
   msm.analysis
   msm.flux
   msm.generation
   msm.ui
