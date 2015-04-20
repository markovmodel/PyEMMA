1.2
----------------------
1.2 is a major new release which offers a load of new and useful functionalities for coordinate
loading, data processing and Markov model estimation and analysis. In a few places we had to change
existing API functions, but we encourage everyone to update to 1.2.

- coordinate package: featurizer can be constructed separately
- coordinate package: new functions for loading data and creating file readers for large trajectories
- coordinate package: all clustering functions were renamed (e.g.: kmeans -> cluster_kmeans). Old function names do still work, but are deprecated
- coordinate package: new pipeline() function for generic data processing pipelines. Using pipelines you can go from data loading, over transformation via TICA or PCA, to clustered data all via stream processing. This avoids having to load large datasets into memory. 
- msm package: markov_model() function creates a MSM object that offers a lot of analysis functions such as spectral analysis, mean first passage times, pcca, calculation of experimental observables, etc.
- msm package: estimate_markov_model() function creates a EstimatedMSM object from data. Offers all functionalities of MSM plus additional functions related to trajectories, such as drawing representative smaples for MSM states
- msm package: Chapman-Kolmogorow test and implied timescales calculation are more robust
- msm package: cktest() and tpt() functions now accept MSM objects as inputs
- various bug fixes

1.1.2 (3-18-2015)
-----------------

- PCCA++ now produces correct memberships (fixes a problem from nonorthonormal eigenvectors)
- Improved Coordinates API documentation (Examples, examples, EXAMPLES)
