Changelog
=========

1.2.2 (27-7-2015)
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


1.2.1 (28-5-2015)
-----------------
- general: Time consuming algorithms now display progressbars (optional).
- general: removed scikit-learn dependency (due to new kmeans impl. Thanks @clonker)
- coordinates package: new and faster implementation of Kmeans (10x faster than scikit-learn).
- coordinates package: allow metrics to be passed to cluster algorithms.
- coordinates package: cache trajectory lengths by default -
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
    - [coordinates.NumPyReader]: during gathering of shapes of all files, none
                                 of them were closed. Thanks @gph82

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
