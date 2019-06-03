
Learn PyEMMA
============

We provide two major sources of learning materials to master PyEMMA, our collection of Jupyter notebook tutorials and
videos of talks given at our annual workshop.

The notebooks are a collection of complete application walk-throughs capturing the most important aspects of building and
analyzing a Markov state model.


Jupyter notebook tutorials
--------------------------

 .. _ref-notebooks:

By means of three different application examples, these notebooks give an overview of following methods:

   * Featurization and MD trajectory input
   * Time-lagged independent component analysis (TICA)
   * Clustering
   * Markov state model (MSM) estimation and validation
   * Computing metastable states and structures, coarse-grained MSMs
   * Hidden Markov Models (HMM)
   * Transition Path Theory (TPT)

These tutorials are part of a LiveCOMS journal article and are up to date with the current PyEMMA release.

You can find the article `here <https://www.livecomsjournal.org/article/5965-introduction-to-markov-state-modeling-with-the-pyemma-software-article-v1-0>`_.

If you find a mistake or have suggestions for improving parts of the tutorial, you can file issues and pull requests
for the contents of both the article and the jupyter notebooks `here <https://github.com/markovmodel/PyEMMA_tutorials>`_.

.. toctree::
   :maxdepth: 1

   tutorials/notebooks/00-pentapeptide-showcase
   tutorials/notebooks/01-data-io-and-featurization
   tutorials/notebooks/02-dimension-reduction-and-discretization
   tutorials/notebooks/03-msm-estimation-and-validation
   tutorials/notebooks/04-msm-analysis
   tutorials/notebooks/05-pcca-tpt
   tutorials/notebooks/06-expectations-and-observables
   tutorials/notebooks/07-hidden-markov-state-models
   tutorials/notebooks/08-common-problems




Workshop video tutorials
------------------------

On our Youtube channel you will find lectures and talks about:

  * Markov state model theory
  * Hidden Markov state models
  * Transition path theory
  * Enhanced sampling
  * Dealing with multi-ensemble molecular dynamics simulations in PyEMMA
  * Useful hints about practical issues...


2018 Workshop
^^^^^^^^^^^^^

.. raw:: html

   <iframe width="100%" height="480px" src="https://www.youtube-nocookie.com/embed/videoseries?list=PLych0HcnzSQLi1CQmxiZig9frLGidF70K"></iframe>


2017 Workshop
^^^^^^^^^^^^^

.. raw:: html

   <iframe width="100%" height="480px" src="https://www.youtube-nocookie.com/embed/videoseries?list=PLych0HcnzSQIBl_AZN5L2cMZvOfyhzjd8"></iframe>

|
|
|

The legacy tutorials (prior version 2.5.5) covering similar aspects and advanced topics can be found here:

.. toctree::
   :maxdepth: 2

   ipython
