.. _ref_install:

============
Installation
============

Recommended way
===============

Use the `Anaconda`_ scientific Python stack to easily install PyEMMA software and
its dependencies.


To install the PyEMMA Python package, you need a few Python package dependencies
like **NumPy** and **SciPy**.

No matter whether you choose the binary or source install method these will be
installed automatically during the setup process.

If you are using the sources to install, you have to note that if your current
Python environment does not contain NumPy and SciPy, those packages will be
built from source too. Building these from source is sometimes tricky, takes a
long time and is error prone - though it is **not** recommended nor supported
by us.


You should either ensure you have Numpy and Scipy installed prior to a source
build of Emma or use a binary installation method, which will install
dependencies automatically.

No matter if you choose binary or source install, you have to ensure the
prequisites in the next section are met or you will most likely encounter
errors.

Binary Packages
===============

Anaconda
~~~~~~~~

It is recommended to use the binary **Anaconda** Python distribution, as it is
easy to install the difficult to build packages NumPy and SciPy under MacOSX
and Windows.

Get it a minimal distribution for Python **2.7** for your operating system from
http://conda.pydata.org/miniconda.html

After setting it up, you can install Emma via the conda package manager from the
`Omnia MD <http://www.omnia.md/>`_ software channel.

::

   conda config --add channels https://conda.binstar.org/omnia
   conda install pyemma

Python Package Index (PyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you do not like Anaconda for some reason you should use the Python package
manager **pip** to install. If you do not have pip, please read its
`install guide <http://pip.readthedocs.org/en/latest/installing.html>`_.


Then make sure pip is enabled to install so called
`wheel <http://wheel.readthedocs.org/en/latest/>`_ packages:

::

   pip install wheel

Then you are able to install binaries if you are have MacOSX or Windows.
Binaries for all flavours of Linux are currently out of our scope - sorry.
Please read howto build from source.

Building from Source
====================
If you are a developer, want to have optimized builds of Emma for your platform
or are using Linux, you want to build it from source. This guide assumes you
have NumPy and SciPy installed.


Prequisites
~~~~~~~~~~~
 * C/C++ compiler
 * Have a valid JAVA_HOME.
 * recent version of Python setuptools


Setuptools
~~~~~~~~~~
It is recommended to upgrade to latest setuptools for a smooth installation
process. Invoke pip to upgrade:

::

    pip install --upgrade setuptools


Building/Installing
~~~~~~~~~~~~~~~~~~~
The build and install process is in one step, as all dependencies are dragged in
via the provided *setup.py* script. So you only need to get the source of Emma
and run it to build Emma itself and all of its dependencies (if not already
supplied) from source.

Recommended for users:

::

   pip install pyemma

Recommended method for developers using GIT:

1. Obtain a clone via

::

   git clone https://github.com/markovmodel/PyEMMA.git

2. install pyemma via

::

   python setup.py develop [--user]

The develop install has the advantage that if only python scripts are being changed
eg. via an pull or a local edit, you do not have to reinstall anything, because
the setup command simply created a link to your working copy.
