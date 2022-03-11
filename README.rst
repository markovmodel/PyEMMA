=====================================
EMMA (Emma's Markov Model Algorithms)
=====================================

.. image:: https://img.shields.io/azure-devops/build/clonker/e16cf5d1-7827-4597-8bb5-b0f7577c73d1/6
   :target: https://dev.azure.com/clonker/pyemma/_build
.. image:: https://img.shields.io/pypi/v/pyemma.svg
   :target: https://pypi.python.org/pypi/pyemma
.. image:: https://anaconda.org/conda-forge/pyemma/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/pyemma
.. image:: https://anaconda.org/conda-forge/pyemma/badges/installer/conda.svg
   :target: https://conda.anaconda.org/conda-forge
.. image:: https://img.shields.io/codecov/c/github/markovmodel/PyEMMA/devel.svg
   :target: https://codecov.io/gh/markovmodel/PyEMMA/branch/devel


What is it?
-----------
PyEMMA (EMMA = Emma's Markov Model Algorithms) is an open source
Python/C package for analysis of extensive molecular dynamics simulations.
In particular, it includes algorithms for estimation, validation and analysis
of:

  * Clustering and Featurization
  * Markov state models (MSMs)
  * Hidden Markov models (HMMs)
  * Multi-ensemble Markov models (MEMMs)
  * Time-lagged independent component analysis (TICA)
  * Transition Path Theory (TPT)

PyEMMA can be used from Jupyter (former IPython, recommended), or by
writing Python scripts. The docs, can be found at
`http://pyemma.org <http://www.pyemma.org/>`__.


Citation
--------
If you use PyEMMA in scientific work, please cite:

    M. K. Scherer, B. Trendelkamp-Schroer, F. Paul, G. Pérez-Hernández,
    M. Hoffmann, N. Plattner, C. Wehmeyer, J.-H. Prinz and F. Noé:
    PyEMMA 2: A Software Package for Estimation, Validation, and Analysis of Markov Models,
    J. Chem. Theory Comput. 11, 5525-5542 (2015)


Installation
------------
If you want to use Miniconda on Linux or OSX, you can run this script to download and install everything::

   curl -s https://raw.githubusercontent.com/markovmodel/PyEMMA/devel/install_miniconda%2Bpyemma.sh | bash

If you have Anaconda/Miniconda installed, use the following::

   conda install -c conda-forge pyemma

With pip::

   pip install pyemma

or install latest devel branch with pip::

   pip install git+https://github.com/markovmodel/PyEMMA.git@devel

For a complete guide to installation, please have a look at the version
`online <http://www.emma-project.org/latest/INSTALL.html>`__ or offline in file
doc/source/INSTALL.rst

To build the documentation offline you should install the requirements with::

   pip install -r requirements-build-doc.txt

Then build with make::

   cd doc; make html


Support and development
-----------------------
For bug reports/suggestions/complaints please file an issue on
`GitHub <http://github.com/markovmodel/PyEMMA>`__.

Or start a discussion on our mailing list: pyemma-users@lists.fu-berlin.de


External Libraries
------------------
* mdtraj (LGPLv3): https://mdtraj.org
* msmtools (LGLPv3): http://github.com/markovmodel/msmtools
* thermotools (LGLPv3): http://github.com/markovmodel/thermotools
