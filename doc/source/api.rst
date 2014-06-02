.. _ref_api:

API Reference
=============

Markov State Models
-------------------
These packages provide functions to estimate, analyze and generate Markov state
models. All public functions accept dense NumPy and SciPy sparse types and distinguish
them automatically, so the optimal underlying solution for a problem is choosen.


.. toctree::
   :maxdepth: 1

   msm.analysis
   msm.estimation
   msm.io
   msm.generation
   
Coordinates
-----------
The *coordinates* package implements common transformations used in Markov state
modeling, like RMSD, TICA etc.

.. toctree::
   :maxdepth: 1
   
   coordinates
