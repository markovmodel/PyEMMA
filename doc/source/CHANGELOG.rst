Changelog
=========

2.5.8 (??-??-????)
-----------------

**New features**:
- coordinates: added contact counting for GroupMinDistanceFeature and ResidueMinDistanceFeature. :pr:`1441`
- plots: added multi cktest support to plot_cktest function. :pr:`1450`


**Fixes**:
- coordinates: fixed a bug in description of sidechain torsion featurizer. :pr:`1451`
- serialization: fixed bug in function which checked for h5 serialization options.
- :code:`n_jobs` is handled consistently, allows only for :code:`None` or positive integers and when
  determined from hardware, falls back to logical number of cpus. :pr:`1488`


2.5.7 (9-24-2019)
-----------------

**New features**:

- msm: added milestone counting to ML-MSM, AugmentedMSM, and BayesianMSM (not effective sampling) based on
  last visited core. Added API infrastructure to include more milestone counting methods. :pr:`1000`


**Fixes**:

- core: fixed a bug occurring with the usage of progress bars in jupyter notebooks (web frontend). :pr:`1434:`


2.5.6 (05-24-2019)
------------------

**New features**:

- all dependencies can seemlessly be installed via pip. :pr:`1412`

**Fixes**:

- NumPy 1.16 made it impossible to change the writeable flag,
  which led to a bug in PyEMMA (CK-test failures with multiple jobs). :pr:`1406`


2.5.5 (02-13-2019)
------------------

For now all future versions will only support Python 3. :pr:`1395`

**New features**:

- msm: for Bayesian MSMs we show an optional progress bar for the counting computation. :pr:`1344`
- msm: ImpliedTimescales allow to only store timescales and its samples for the purpose of saving memory. :pr:`1377`
- double_well_discrete: allows the setting of a random seed :pr:`1388`


**Fixes**:

- msm: fix connected sets for Bayesian MSMs in case the mincount connectivity (ergodic cutoff) parameter truncated
  the count matrix. :pr:`1343`
- plots: fix vmin, vmax keyword arguments for plot_contour(). :pr:`1376`
- coordinates: forcefully enable checking of coordinates data streaming for invalid (not initialized) data. :pr:`1384`
- coordinates: for sake of faster strided MD data reading, we now require a version of MDTraj >= 1.9.2 :pr:`1391`
- coordinates: VAMP: changed default projection vector from right to left, since only the left singular functions induce
  a kinetic map wrt. the conventional forward propagator, while the right singular functions induce
  a kinetic map wrt. the backward propagator. :pr:`1394`
- coordinates: VAMP: fixed variance cutoff to really include as many dimensions to meet subspace variance criterion. :pr:`1397`


**Contributors**:

- :user:`marscher`
- :user:`fabian-paul`
- :user:`brookehus`
- :user:`thempel`


2.5.4 (07-20-2018)
------------------

**New features**:

- plots: allow zorder parameter via **kwargs in plot_density(), plot_free_energy(), plot_contour(), and plot_state_map() :pr:`1336`
- plots: allow colorbar orientation via the cbar_orientation parameter in plot_density(), plot_free_energy(), plot_contour(), and plot_state_map() :pr:`1338`

**Fixes**:

- plots: added missing parameter ncontours=100 to plot_state_map() :pr:`1331`
- msm: Chapman Kolmogorov tests (CK-tests) are now computed using multiple processes by default. :pr:`1330`
- coordinates: do not show a progress bar for creating the data array, if data comes from memmory. :pr:`1339`
- plots: maks zero-counts in logscale feature histograms. :pr:`1340`


**Contributors**:

- :user:`cwehmeyer`
- :user:`marscher`


2.5.3 (06-28-2018)
------------------

**New features**:

- plots: new functions plot_density(), plot_state_map(), and plot_contour() :pr:`1317`

**Fixes**:

- base: restored VAMP estimators reset the diagonalization flag, which led to recomputing expensive
  operations. :pr:`1294`
- base: require at least tqdm >= 4.23, because of an API change. :pr:`1292,1293`
- coordinates: fix closing progress bar of kmeans. :pr:`1315`
- coordinates: method output_type of DataSources now returns an instance instead of a class. :pr:`1315`
- coordinates: During processing the actual data is always being checked for invalid values like NaN and infinity. :pr:`1315`
- coordinates: Use IO-efficient time shifted iterator for chunksize 0 (whole trajectories). :pr:`1315`
- coordinates: fixed a bug in internal lengths calculation of FragmentedTrajectoryReader, which led to preliminary
  stopping of iteration. This was only affected by very rare side-conditions. :pr:`1315`
- coordinates: fixed a bug in csv reader, which led to preliminary stopping of iteration. :pr:`1300,1315`
- msm: fixed minor bug in ImpliedTimescales, where all models got recomputed for extended lag time array. :pr:`1294`
- msm: fixed serialization of BayesianHMSM, if initialized with a ML-HMSM. :pr:`1283`
- msm: fixed inconsistent submodel behavior in HMSM and BayesianHMSM. :pr:`1323`
- msm: fixed missing "has_errors" attribute after deserialization. :pr:`1295,1296`
- msm: use stationary distribution estimate of msmtools during MSM estimation. :pr:`1159`
- msm: reset eigenvalue decomposition, if a new transition matrix is encapsulated in the model. This led to weird
  results in CK-test. :pr:`1301,1302`
- plots: fixed minor bug in plot_network (state_labels=None would not work). :pr:`1306`
- plots: refactored plots2d to remove inappropriate pylab/gca() usage, allow more figure construction control :pr:`1317`
- plots: refactored plots1d to remove inappropriate pylab/gca() usage :pr:`1317`


**Contributors**:

- :user:`chwehmeyer`
- :user:`clonker`
- :user:`jeiros`
- :user:`marscher`
- :user:`ppxasjsm`
- :user:`thempel`
- :user:`yanhuaouyang`

2.5.2 (04-10-2018)
------------------

**New features**:

- coordinates: added Nystroem-TICA, which uses sparse sampling to approximate the input space. :pr:`1261,1273`
- plots: added multi-dimensional stacked histogram plot function. :pr:`1264`

**Fixes**:

- msm: Chapman Kolmogorov validator ensures there are no side effects on the tested model. :pr:`1255`
- datasets: Fix default values for kT to ensure integrator produces sane values. :pr:`1272,1275`
- coordinates: fixed fixed handling of default chunksize. :pr:`1284`


2.5.1 (02-17-2018)
------------------

Quick fix release to repair chunking in the coordinates package.

**Fixes**:

- msm: fix bug in ImpliedTimescales, which happened when an estimation failed for a given lag time. :pr:`1248`
- coordinates: fixed handling of default chunksize. :pr:`1247,1251`, :pr:`1252`
- base: updated pybind to 2.2.2. :pr:`1249`


2.5 (02-09-2018)
----------------

As of this version the usage of Python 2.7 is officially deprecated. Please upgrade
your Python installation to at least version 3.5 to catch future updates.

**New features**:

- base: most Estimators and Models in msm, thermo and coordinates packages can be saved to disk now.
  Multiple models/estimators can be stored in the same file, which uses HDF5 as backend. :pr:`849, 867, 1155, 1200, 1205`
- msm: Added Augmented Markov Models. A way to include averaged experimental
  data into estimation of Markov models from molecular simulations. The method is described in [1]. :pr:`1111`
- msm: Added mincount_connectivity argument to MSM estimators. This option enables to omit counts below
  a given threshold. :pr:`1106`
- coordinates: selection based features allow alignment to a reference structure. :pr:`1184`
- coordinates: two new center of mass features: ResidueCOMFeature() and GroupCOMFeature()
- coordinates: new configuration variable 'default_chunksize' can be set to limit the size of a fragmented
  extracted per iteration from a data source. This is invariant to the dimension of data sets. :pr:`1190`
- datasets: added Prinz potential (quadwell). :pr:`1226`
- coordinates: added VAMP estimator. :pr:`1237`
- coordinates: added method 'write_to_hdf5' for easy exporting streams to HDF5. :pr:`1242`

- References:

  [1] Olsson S, Wu H, Paul F, Clementi C, Noe F: Combining experimental and simulation data of molecular
      processes via augmented Markov models. PNAS 114, 8265-8270 (2017).

**Fixes**:

- datasets: fixed get_multi_temperature_data and get_umbrella_sampling_data for Python 3. :pr:`1102`
- coordinates: fixed StreamingTransformers (TICA, Kmeans, etc.) not respecting the in_memory flag. :pr:`1112`
- coordinates: made TrajectoryInfoCache more fail-safe in case of concurrent processes. :pr:`1122`
- msm: fix setting of dt_model for BayesianMSM. This bug led to wrongly scaled time units for mean first passage times,
  correlation and relaxation times as well for timescales for this estimator. :pr:`1116`
- coordinates: Added the covariance property of time-lagged to CovarianceLagged. :pr:`1125`
- coordinates: clustering code modernized in C++ with pybind11 interface. :pr:`1142`
- variational: covartools code modernized in C++ with pybind11 interface. :pr:`1147`
- estimators: n_jobs setting does not look for OMP_NUM_THREADS, but for PYEMMA_NJOBS and SLURM_CPUS_ON_NODE to avoid
  multiplying OpenMP threads with PyEMMA processes. On SLURM the number of allocated cores is used.
  If nothing is set, the physical cpu count is considered.
- msm: calling score_cv does not modify the object anymore. :pr:`1178`
- base:estimator: fixed signature of fit function for compatability with scikit-learn. :pr:`1193`
- coordinates: assign_to_centers now handles stride argument again. :pr:`1190`


2.4 (05-19-2017)
----------------

**New features**:

- msm: variational scores for model selection of MSMs. The scores are based on the variational
  approach for Markov processes [1, 2] and can be employed for both reversible and non-reversible
  MSMs. Both the Rayleigh quotient as well as the kinetic variance [3] and their non-reversible
  generalizations are available. The scores are implemented in the `score` method of the MSM
  estimators `MaximumLikelihoodMSM` and `OOMReweightedMSM`. Rudimentary support for Cross-validation
  similar as suggested in [4] is implemented in the `score_cv` method, however this is currently
  inefficient and will be improved in future versions. :pr:`1093`

- config: Added a lot of documentation and added `mute` option to silence PyEMMA (almost completely).

- References:
    [1] Noe, F. and F. Nueske: A variational approach to modeling slow processes
        in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655 (2013).
    [2] Wu, H and F. Noe: Variational approach for learning Markov processes
        from time series data (in preparation).
    [4] Noe, F. and C. Clementi: Kinetic distance and kinetic maps from molecular
        dynamics simulation. J. Chem. Theory Comput. 11, 5002-5011 (2015).
    [3] McGibbon, R and V. S. Pande: Variational cross-validation of slow
        dynamical modes in molecular kinetics, J. Chem. Phys. 142, 124105 (2015).

- coordinates:
   - kmeans: allow the random seed used for initializing the centers to be passed. The prior behaviour
     was to init the generator by time, if fixed_seed=False. Now bool and int can be passed. :pr:`1091`

- datasets:
   - added a multi-ensemble data generator for the 1D asymmetric double well. :pr:`1097`

**Fixes**:

- coordinates:
  - StreamingEstimators: If an exception occurred during flipping the `in_memory` property,
    the state is not updated. :pr:`1096`
  - Removed deprecated method parametrize. Use estimate or fit for now. :pr:`1088`
  - Readers: nice error messages for file handling errors (which file caused the error). :pr:`1085`
  - TICA: raise ZeroRankError, if the input data contained only constant features. :pr:`1055`
  - KMeans: Added progress bar for collecting the data in pre-clustering phase. :pr:`1084`

- msm:
  - ImpliedTimescales estimation can be interrupted (strg+c, stop button in Jupyter notebooks). :pr:`1079`

- general:
  - config: better documentation of the configuration parameters. :pr:`1095`


2.3.2 (2-19-2017)
-----------------

**New features**:

thermo:

- Allow for periodicity in estimate_umbrella_sampling().
- Add *_full_state getter variants to access stationary properties on the full set of states
  instead of the active set.

**Fixes**:

coordinates:

- [TICA] fixed regularization of timescales for the non-default feature **commute_map**. :pr:`1037,1038`

2.3.1 (2-6-2017)
----------------

**New features**:

- msm:
   - ImpliedTimescales: enable insertion/removal of lag times.
     Avoid recomputing existing models. :pr:`1030`

**Fixes**:

- coordinates:
   - If Estimators supporting streaming are used directly, restore previous behaviour. :pr:`1034`
     Note that estimators used directly from the API were not affected.


2.3 (1-6-2017)
--------------

**New features**:

- coordinates:
   - tica: New option "weights". Can be "empirical", which does the same as before,
     or "koopman", which uses the re-weighting procedure from [1] to compute equi-
     librium covariance matrices. The user can also supply his own re-weighting me-
     thod. This must be an object that possesses a function weights(X), that assigns
     a weight to every time-step in a trajectory X. :pr:`1007`
   - covariance_lagged: This new method can be used to compute covariance matrices
     and time-lagged covariance matrices between time-series. It is also possible
     to use the re-weighting method from [1] to compute covariance matrices in equi-
     librium. This can be triggered by the option "weights", which has the same spe-
     cifications as in tica. :pr:`1007`

- msm:
   - estimate_markov_model: New option "weights". Can be empirical, which does the
     same as before, or "oom", which triggers a transition matrix estimator based
     on OOM theory to compute an equilibrium transition matrix from possibly non-
     equilibrium data. See Ref. [2] for details. :pr:`1012,1016`
   - timescales_msm: The same change as in estimate_markov_model. :pr:`1012,1016`
   - TPT: if user provided sets A and B do not overlap (no need to split), preserve order of user states. :pr:`1005`

- general: Added an automatic check for new releases upon import. :pr:`986`

- References:
   [1] Wu, H., Nueske, F., Paul, F., Klus, S., Koltai, P., and Noe, F. 2017. Bias reduced variational
        approximation of molecular kinetics from short off-equilibrium simulations. J. Chem. Phys. (submitted),
        https://arxiv.org/abs/1610.06773.
   [2] Nueske, F., Wu, H., Prinz, J.-H., Wehmeyer, C., Clementi, C., and Noe, F. 2017. Markov State Models from
        short non-Equilibrium Simulations - Analysis and Correction of Estimation Bias. J. Chem. Phys.
        (submitted).


**Fixes**:

- coordinates:
   - kmeans: fixed a rare bug, which led to a segfault, if NaN is contained in input data. :pr:`1010`
   - Featurizer: fix reshaping of AnglesFeature. :pr:`1018`. Thanks :user:`RobertArbon`

- plots: Fix drawing into existing figures for network plots. :pr:`1020`


2.2.7 (10-21-16)
----------------

**New features**:

- coordinates:
   - for lag < chunksize improved speed (50%) for TICA. :pr:`960`
   - new config variable "coordinates_check_output" to test for "NaN" and "inf" values in
     iterator output for every chunk. The option is disabled by default. It gives insight
     during debugging where faulty values are introduced into the pipeline. :pr:`967`


**Fixes**:

- coordinates:
   - save_trajs, frames_from_files: fix input indices checking. :pr:`958`
   - FeatureReader: fix random access iterator unitcell_lengths scaling.
     This lead to an error in conjunction with distance calculations, where
     frames are collected in a random access pattern. :pr:`968`
- msm: low-level api removed (use msmtools for now, if you really need it). :pr:`550`

2.2.6 (9-23-16)
---------------

**Fixes**:

- msm: restored old behaviour of updating MSM parameters (only update if not set yet).
  Note that this bug was introduced in 2.2.4 and leads to strange bugs, eg. if the MSM estimator
  is passed to the Chapman Kolmogorov validator, the reversible property got overwritten.
- coordinates/TICA: Cast the output of the transformation to float. Used to be double. :pr:`941`
- coordinates/TICA: fixed a VisibleDeprecationWarning. :pr:`941`. Thanks :user:`stefdoerr`

2.2.5 (9-21-16)
---------------

**Fixes**:

- msm: fixed setting of 'reversible' attribute. :pr:`935`

2.2.4 (9-20-16)
---------------

**New features**:

- plots: network plots can now be plotted using a given Axes object.
- thermo: TRAM supports the new parameter equilibrium which triggers a TRAMMBAR estimation.
- thermo: the model_active_set and msm_active_set attributes in estimated MEMMs is deprecated; every
  MSM in models now contains its own active_set.
- thermo: WHAM and MBAR estimations return MultiThermModel objects; return of MEMMs is reserved for
  TRAM/TRAMMBAR/DTRAM estimations.

**Fixes**:

- coordinates: MiniBatchKmeans with MD-data is now memory efficient
  and successfully converges. It used to only one batch during iteration. :pr:`887` :pr:`890`
- coordinates: source and load function accept mdtraj.Trajectory objects to extract topology. :pr:`922`. Thanks :user:`jeiros`
- base: fix progress bars for modern joblib versions.
- plots: fix regression in plot_markov_model with newer NumPy versions :pr:`907`. Thanks :user:`ghoti687.`
- estimation: for n_jobs=1 no multi-processing is used.
- msm: scale transition path times by time unit of MSM object in order to get
  physical time scales. :pr:`929`

2.2.3 (7-28-16)
---------------

**New features**:

- thermo: added MBAR estimation

**Fixes**:

- coordinates: In case a configuration directory has not been created yet, the LRU cache
  of the TrajInfo database was failed to be created. :pr:`882`


2.2.2 (7-14-16)
---------------

**New features**:

- coordinates: SQLite backend for trajectory info data. This enables fast access to this data
  on parallel filesystems where multiple processes are writing to the database. This greatly
  speeds ups reader construction and enables fast random access for formats which usually do not
  support it. :pr:`798`
- plots: new optional parameter **arrow_label_size** for network plotting functions to use a custom
  font size for the arrow labels; the default state and arrow label sizes are now determined by the
  matplotlib default. :pr:`858`
- coordinates: save_trajs takes optional parameter "image_molecules" to correct for broken
  molecules across periodic boundary conditions. :pr:`841`

**Fixes**:

- coordinates: set chunksize correctly. :pr:`846`
- coordinates: For angle features it was possible to use both cossin=True and deg=True, which
  makes not sense. :pr:`857`
- coordinates: fixed a memory error in kmeans clustering which affected large data sets (>=64GB). :pr:`839`
- base: fixed a bug in ProgressReporter (_progress_force_finish in stack trace). :pr:`869`
- docs: fixed a lot of docstrings for inherited classes both in coordinates and msm package.


2.2.1 (6-21-16)
---------------

**Fixes**:

- clustering: fixed serious bug in **minRMSD** distance calculation, which led to
  lots of empty clusters. The bug was introduced in version 2.1. If you used
  this metric, please re-assign your trajectories. :pr:`825`
- clustering: fixed KMeans with minRMSD metric. :pr:`814`
- thermo: made estimate_umbrella_sampling more robust w.r.t. input and fixed doumentation. :pr:`812` :pr:`827`
- msm: low-level api usage deprecation warnings only show up when actually used.

2.2 (5-17-16)
-------------

**New features**:

- thermo: added TRAM estimation.
- thermo: added plotting feature for implied timescales.
- thermo: added Jupyter notebook examples: :ref:`ref-notebooks`.
- thermo: show convergence progress during estimation.

**Fixes**:

- clustering: fix parallel cluster assignment with minRMSD metric.
- base: during estimation the model was accessed in an inappropriate way,
  which led to the crash "AttributeError: object has no attribute '_model'" :pr:`764`.
- coordinates.io: fixed a bug when trying to pyemma.coordinates.load certain MD formats.
  The iterator could have returned None in some cases :pr:`790`.
- coordiantes.save_traj(s): use new backend introduced in 2.1, speed up for non random
  accessible trajectory formats like XTC. Avoids reading trajectory info for files not
  being indexed by the input mapping. Fixes :pr:`788`.


2.1.1 (4-18-2016)
-----------------
Service release. Fixes some

**New features**:

- clustering: parallelized clustering assignment. Especially useful for expensive to
  compute metrics like minimum RMSD. Clustering objects now a **n_jobs** attribute
  to set the desired number of threads. For a high job number one should use a
  considerable high chunk size as well.

**Fixes**:

- In parallel environments (clusters with shared filesystem) there will be no
  crashes due to the config module, which tried to write files in users home
  directory. Config files are optional by now.


2.1 (3-29-2016)
---------------

**New features**:

- thermo package: calculate thermodynamic and kinetic quantities from multi-ensemble data

  - Added estimators (WHAM, DTRAM) for multi-ensemble MD data.
  - Added API functions to handle umbrella sampling and multi-temperature MD data.

- msm/hmsm:

  - Maximum likelihood estimation can deal with disconnected hidden transition
    matrices. The desired connectivity is selected only at the end of the
    estimation (optionally), or a posteriori.
  - Much more robust estimation of initial Hidden Markov model.
  - Added option stationary that controls whether input data is assumed to be
    sampled from the stationary distribution (and then the initial HMM
    distribution is taken as the stationary distribution of the hidden
    transition matrix), or not (then it's independently estimated using the EM
    standard approach). Default: stationary=False. This changes the default
    behaviour w.r.t. the previous version, but in a good way: Now the
    maximum-likelihood estimator always converges. Unfortunately that also
    means it is much slower compared to previous versions which stopped
    without proper convergence.
  - Hidden connectivity: By default delivers a HMM with the full hidden
    transition matrix, that may be disconnected. This changes the default
    behaviour w.r.t. the previous version. Set connectivity='largest' or
    connectivity='populous' to focus the model on the largest or most populous
    connected set of hidden states
  - Provides a way to measure connectivity in HMM transition matrices: A
    transition only counts as real if the hidden count matrix element is
    larger than mincount_connectivity (by default 1 over the number of hidden
    states). This seems to be a much more robust metric of real connectivity
    than MSM count matrix connectivity.
  - Observable set: If HMMs are used for MSM coarse-graining, the MSM active
    set will become the observed set (as before). If a HMM is estimated
    directly, by default will focus on the nonempty set (states with nonzero
    counts in the lagged trajectories). Optionally can also use the full set
    labels - in this case no indexing or relabelling with respect to the
    original clustered data is needed.
  - Hidden Markov Model provides estimator results (Viterbi hidden
    trajectories, convergence information, hidden count matrix). Fixes :pr:`528`
  - BayesianHMSM object now accepts Dirichlet priors for transition matrix and
    initial distribution. Fixes :pr:`640` (general, not only for HMMs) by allowing
    estimates at individual lag times to fail in an ImpliedTimescales run
    (reported as Warnings).

- coordinates:
    - Completely re-designed class hierachy (user-code/API unaffected).
    - Added trajectory info cache to avoid re-computing lengths, dimensions and
      byte offsets of data sets.
    - Random access strategies supported (eg. via slices).
    - FeatureReader supports random access for XTC and TRR (in conjunction with mdtraj-1.6).
    - Re-design API to support scikit-learn interface (fit, transform).
    - Pipeline elements (former Transformer class) now uses iterator pattern to
      obtain data and therefore supports now pipeline trees.
    - pipeline elements support writing their output to csv files.
    - TICA/PCA uses covartools to estimate covariance matrices:
        + This now saves one pass over the data set.
        + Supports sparsification data on the fly.

**Fixes**:

- HMM Chapman Kolmogorov test for large datasets :pr:`636`.
- Progressbars now auto-hide, when work is done.


2.0.4 (2-9-2016)
----------------
Patch release to address DeprecationWarning flood in conjunction with Jupyther notebook.

2.0.3 (1-29-2016)
-----------------

**New features**:

- msm: added keyword "count_mode" to estimate_markov_model, to specify the way
  of counting during creation of a count matrix. It defaults to the same behaviour
  like prior versions (sliding window). New options:

  - 'effective': Uses an estimate of the transition counts that are
     statistically uncorrelated. Recommended when used with a Bayesian MSM.
  - 'sample': A trajectory of length T will have T/tau counts at time indices
     0 -> tau, tau -> 2 tau, ..., T/tau - 1 -> T

- msm: added possibility to constrain the stationary distribution for BayesianMSM
- coordinates: added "periodic" keyword to features in Featurizer to indicate a
  unit cell with periodic boundary conditions.
- coordinates: added "count_contacts" keyword to Featurizer.add_contacts() method
  to count formed contacts instead of dimension of all possible contacts.
- logging: pyemma.log file will be rotated after reaching a size of 1 MB

**Fixes**:

- logging: do not replace existing loggers anymore. Use hierarchical logging (all loggers
  "derive" from 'pyemma' logger. So log levels etc. can be manipulated by changing this
  new 'pyemma' root logger.
- some deprecation warnings have been fixed (IPython and Python-3.5 related).

2.0.2 (11-9-2015)
-----------------

**New features**:

- coordinates: added Sparsifier, which detects constant features in data stream
  and removes them for further processing.
- coordinates: cache lengths of NumPy arrays
- coordinates: clustering.interface new methods index_clusters and sample_indexes_by_cluster
- coordinates: featurizer.add_contacts has new threshold value of .3 nm
- coordinates: featurizer.pairs gets opt arg excluded_neighbors (default (=0) is unchanged)
- coordinates: featurizer.describe uses resSeq instead of residue.index
- plots: network plots gets new arg state_labels, arg state_colors extended, textkwargs added
- plots: timescale plot accepts different units for x,y axes
- logging: full-feature access to Python logging system (edit logging.yml in .pyemma dir)

**Fixes**:

- Upon import no deprecation warning (about acf function) is shown.
- coordinates: chunksize attribute moved to readers (no consequence for user-scripts)
- coordinates: fixed bug in parallel evaluation of Estimators, when they have active loggers.
- documentation fixes

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
- general: removed scikit-learn dependency (due to new kmeans impl. Thanks :user:`clonker)`
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
