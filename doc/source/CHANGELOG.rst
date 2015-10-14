Changelog
=========

2.0.1 (to be released)
----------------------

** New features**:

- coordinates: added Sparsifier, which detects constant features in data stream
  and removes them for further processing. 


2.0.1 (9-3-2015)
----------------
Urgent bug fix: reading other formats than XTC was not possible in coordinates
pipeline. This bug has been introduced into 2.0, prior versions were not affected.

2.0 (9-1-2015)
--------------
2.0 is a major release offering several new features and a major internal
reorganization of the code.

**New features**:

- coordinates: Featurizer new features: ResidueMinDistanceFeature and GroupMinDistanceFeature.
- coordinates: PCA and TICA use a default variance cutoff of 95%.
- coordinates: TICA is scaled to produce a kinetic map by default.
- coordinates: TICA eigenvalues can be used to calculate timescales.
- coordinates: new MiniBatchKmeans implementation.
- coordinates: Early termination of pipeline possible (eg. max_clusters reached).
- coordinates: random access of input through pipeline via indices.
- msm: Estimator for Bayesian Markov state models.
- msm: MSMs can be systematically coarse-grained to few-state models
- msm: Estimators for discrete Hidden Markov Models (HMMs) and Bayesian Hidden Markov models (BHMMs).
- msm: SampledModels, e.g. generated from BayesianMSM or BayesianHMM allow statistics
  (means, variances, confidence intervals) to be computed for all properties of MSMs and HMMs.
- msm: Generalized Chapman-Kolmogorov test for both MSM and HMM models
- plots: plotting functions for Chapman-Kolmogorov tests and 2D free energy surfaces.
- plots: more flexible network plots.

**Documentation**:

- One new application-based ipython notebooks and three new methodological ipython notebooks
  are provided. All Notebooks and most of the data are provided for download at pyemma.org.
- Many improvements in API documentation.

**Code architecture**:

- Object structure is more clear, general and extensible. We have three main
  class types: Estimators, Transformers and Models. Estimators (e.g. MaximumLikelihoodMSM)
  read data and produce a Transformer or a Model. Transformers (e.g. TICA) can be employed in
  order to transform input data into output data (e.g. dimension reduction). Models
  (e.g. MSM) can be analyzed in order to compute molecular quantities of interest, such
  as equilibrium probabilities or transition rates.
- Estimators and Transformers have basic compatibility with scikit-learn objects.
- Code for low-level msm functions (msm.analysis, msm.estimation, msm.generation, msm.flux) has
  been relocated to the subsidiary package msmtools (github.com/markovmodel/msmtools). msmtools is
  part of the PyEMMA distribution but can be separately installed without depending on
  PyEMMA in order to facilitate further method development.
- Removed deprecated functions from 1.1 that were kept during 1.2


1.2.2 (7-27-2015)
-----------------
- msm estimation: new fast transition matrix sampler
- msm estimation: new feature "auto-sparse": automatically decide which datatype 
  to use for transition matrix estimation.
- coordinates package: kinetic map feature for TICA (arXiv:1506.06259 [physics.comp-ph])
- coordinates package: better examples for API functions.
- coordinates package: cluster assignment bugfix in parallel environments (OpenMP). 
- coordinates package: added cos/sin transformations for angle based features to
  featurizer. This is recommended for PCA/TICA transformations.
- coordinates package: added minimum RMSD feature to featurizer.
- coordinates package: Regular space clustering terminates early now, when it reaches
  max_clusters cutoff.
- plots package: use Fruchterman Reingold spring algorithm to calculate positions
  in network plots.
- ipython notebooks: new real-world examples, which show the complete workflow
- general: made all example codes in documentation work.


1.2.1 (5-28-2015)
-----------------
- general: Time consuming algorithms now display progressbars (optional).
- general: removed scikit-learn dependency (due to new kmeans impl. Thanks @clonker)
- coordinates package: new and faster implementation of Kmeans (10x faster than scikit-learn).
- coordinates package: allow metrics to be passed to cluster algorithms.
- coordinates package: cache trajectory lengths by default
                       (uncached led to 1 pass of reading for non indexed (XTC) formats).
  This avoids re-reading e.g XTC files to determine their lengths.
- coordinates package: enable passing chunk size to readers and pipelines in API.
- coordinates package: assign_to_centers now allows all supported file formats as centers input.
- coordinates package: save_traj(s) now handles stride parameter.
- coordinates package: save_traj    now accepts also lists of files as an input 
  In this case, an extra parameter topfile has to be parsed as well.
- plots package: added functions to plot flux and msm models.
- Bugfixes:

   - [msm.MSM.pcca]: coarse-grained transition matrix corrected
   - [msm.generation]: stopping states option fixed
   - [coordinates.NumPyReader]: during gathering of shapes of all files, none of them were closed.

1.2 (4-14-2015)
---------------
1.2 is a major new release which offers a load of new and useful functionalities
for coordinate loading, data processing and Markov model estimation and analysis. 
In a few places we had to change existing API functions, but we encourage
everyone to update to 1.2.

- coordinate package: featurizer can be constructed separately
- coordinate package: new functions for loading data and creating file readers
  for large trajectories
- coordinate package: all clustering functions were renamed 
  (e.g.: kmeans -> cluster_kmeans). Old function names do still work, but are deprecated
- coordinate package: new pipeline() function for generic data processing pipelines.
  Using pipelines you can go from data loading, over transformation via TICA or PCA,
  to clustered data all via stream processing. This avoids having to load large 
  datasets into memory.
- msm package: markov_model() function creates a MSM object that offers a lot 
  of analysis functions such as spectral analysis, mean first passage times, 
  pcca, calculation of experimental observables, etc.
- msm package: estimate_markov_model() function creates a EstimatedMSM object
  from data. Offers all functionalities of MSM plus additional functions related
  to trajectories, such as drawing representative smaples for MSM states
- msm package: Chapman-Kolmogorow test and implied timescales calculation are more robust
- msm package: cktest() and tpt() functions now accept MSM objects as inputs
- various bug fixes

1.1.2 (3-18-2015)
-----------------

- PCCA++ now produces correct memberships (fixes a problem from nonorthonormal eigenvectors)
- Improved Coordinates API documentation (Examples, examples, EXAMPLES)
