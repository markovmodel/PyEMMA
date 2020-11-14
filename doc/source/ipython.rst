 .. _ref-notebooks_legacy:


Legacy Jupyter Notebook Tutorials
=================================

These Jupyter (http://jupyter.org) notebooks show the usage of
the PyEMMA API in action and also describe the workflow of Markov model
building.

You can download a copy of all notebooks and most of the used data
`here <https://github.com/markovmodel/PyEMMA_IPython/archive/master.zip>`_.
Note that the trajectory of the D.E. Shaw BPTI simulation trajectory is not included
in this archive, since we're not permitted to share this data. Thus the corresponding
notebooks can't be run without obtaining the simulation trajectory independently.

Application walkthroughs
------------------------

.. toctree::
   :maxdepth: 1

   legacy-notebooks/applications/pentapeptide_msm/pentapeptide_msm
   legacy-notebooks/applications/bpti_msm/MSM_BPTI

By means of application examples, these notebooks give an overview of following methods:

   * Featurization and MD trajectory input
   * Time-lagged independent component analysis (TICA)
   * Clustering
   * Markov state model (MSM) estimation and validation
   * Computing Metastable states and structures, coarse-grained MSMs
   * Hidden Markov Models (HMM)
   * Transition Path Theory (TPT)


Methods
-------

In this section we will give you in-depth tutorials on specific methods or concepts.

.. toctree::
   :maxdepth: 1

   legacy-notebooks/methods/feature_selection/feature_selection
   legacy-notebooks/methods/model_selection_validation/model_selection_validation
   legacy-notebooks/methods/tpt/tpt
   legacy-notebooks/methods/amm/augmented_markov_model_walkthrough
   legacy-notebooks/methods/storage/storage


Multi-Ensemble Markov state models
----------------------------------

.. toctree::
   :maxdepth: 1

   legacy-notebooks/methods/multi_ensemble/doublewell/PyEMMA.thermo.estimate_multi_temperatur_-_asymmetric_double_well
   legacy-notebooks/methods/multi_ensemble/doublewell/PyEMMA.thermo.estimate_umbrella_sampling_-_asymmetric_double_well

Estimation with fixed equilibrium distribution
----------------------------------------------

.. toctree::
   :maxdepth: 1

   legacy-notebooks/methods/msm_with_given_equilibrium/doublewell/doublewell
   legacy-notebooks/methods/msm_with_given_equilibrium/alanine/alanine
