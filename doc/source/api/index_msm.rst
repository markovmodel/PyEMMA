.. _msm_api:

Markov State Models API
=======================
The *msm* package provides functions to estimate, analyze and generate
Markov state models. All public functions accept dense NumPy and SciPy
sparse types and distinguish them automatically, so the optimal
underlying solution for a problem is choosen.


.. automodule:: pyemma.msm

.. toctree::
   :maxdepth: 1
   
MSM functions
-------------

.. toctree::
   :maxdepth: 1
   
   msm.io
   msm.estimation
   msm.analysis
   msm.flux
   msm.generation
   msm.ui
