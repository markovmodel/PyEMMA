=====================================
EMMA (Emma's Markov Model Algorithms)
=====================================

.. image:: https://travis-ci.org/markovmodel/PyEMMA.svg?branch=devel
   :target: https://travis-ci.org/markovmodel/PyEMMA
.. image:: https://badge.fury.io/py/pyemma.svg
   :target: https://pypi.python.org/pypi/pyemma
.. image:: https://pypip.in/d/pyemma/badge.svg
   :target: https://pypi.python.org/pypi/pyemma
.. image:: https://binstar.org/xavier/binstar/badges/downloads.svg
   :target: https://binstar.org/omnia/pyemma
.. image:: https://binstar.org/omnia/pyemma/badges/installer/conda.svg
   :target: https://conda.binstar.org/omnia
.. image:: https://coveralls.io/repos/markovmodel/PyEMMA/badge.svg?branch=devel
   :target: https://coveralls.io/r/markovmodel/PyEMMA?branch=devel

What is it?
-----------
EMMA is an open source collection of algorithms implemented mostly in
`NumPy <http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org>`_ to analyze
trajectories generated from any kind of simulation (e.g. molecular
trajectories) via Markov state models (MSM).

It provides APIs for estimation and analyzing MSM and various utilities to
process input data (clustering, coordinate transformations etc). For
documentation of the API, please have a look at the sphinx docs in doc
directory or `online <http://pythonhosted.org/pyEMMA/api/index.html>`__.

For some examples on how to apply the software, please have a look in the
ipython directory, which shows the most common use cases as documentated
IPython notebooks.

Installation
------------
With pip::
 
     pip install pyemma

with conda::

     conda install -c https://conda.binstar.org/omnia pyemma


or install latest devel branch with pip::

     pip install git+https://github.com/markovmodel/PyEMMA.git@devel

For a complete guide to installation, please have a look at the version 
`online <http://pythonhosted.org/pyEMMA/INSTALL.html>`__ or offline in file
doc/source/INSTALL.rst

To build the documentation offline you should install the requirements with::
   
   pip install -r requirements-build-doc.txt

Then build with make::

   cd doc; make html

Support and development
-----------------------
For bug reports/sugguestions/complains please file an issue on 
`GitHub <http://github.com/markovmodel/PyEMMA>`__.

Or start a discussion on our mailing list: pyemma-users@lists.fu-berlin.de


External Libraries
------------------
* mdtraj (LGPLv3): https://mdtraj.org

