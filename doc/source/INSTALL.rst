.. _ref_install:

============
Installation
============

To install the Emma Python package, you need a few Python package dependencies
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

Java/Python Bridge
==================
JPype will be installed automatically, you only need to ensure, you have a
working Java runtime (JRE). For the Java/Python bridge provided by **JPype**
you need a recent Java Runtime (>= 1.6), which is already provided by most
platforms. You are strongly encouraged to set a environment variable
"JAVA_HOME" to point to your desired Java installation. If you are able to
execute *java* from the command line, you are already done. Otherwise you
should set up JAVA_HOME or PATH to enable JPype to find the runtime.

There exists a guide at
`Java.com <https://www.java.com/en/download/help/path.xml>`_ describing the
setting of a **PATH** environment variable.


Setting up a correct JAVA_HOME
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is especially useful if you have
multiple versions of Java and want to force JPype to use a given one.

.. To locate all Java installations on your system, you may try this snippet:
.. .. code-block:: python

   import os.path                                                            
   import sys                                                                     
                                                                                   
   if sys.platform == 'darwin':                                                    
      libfile = 'libjvm.dylib'                                                     
   elif sys.platform in ('win32', 'cygwin'):                                       
      libfile = 'libjvm.dll'                                                       
   else:                                                                           
      libfile = 'libjvm.so'                                                        
   for root, dirs, files in os.walk("/"):                                          
       if libfile in files:                                                        
          print "java found in %s" % root

.. This will print all possible values for JAVA_HOME

Lets assume you have Oracles Java Virtual Machine installed under:
`/usr/lib/jvm/java-8-oracle`. This is just an example, do not try to copy paste
this - it will fail.

So you would like to add the following in your ".bashrc" file (or your prefered
way to set environment variables).

.. code-block:: bash

   export JAVA_HOME=/usr/lib/jvm/java-8-oracle

To test if this is correct, try to execute the java command from your new
JAVA_HOME:

.. code-block:: bash

   $JAVA_HOME/bin/java -version

It should output something like this:

::

   java version "1.8.0_20"
   Java(TM) SE Runtime Environment (build 1.8.0_20-b26)
   Java HotSpot(TM) 64-Bit Server VM (build 25.20-b23, mixed mode)
   
If the command is not found, the JAVA_HOME does not point to the correct
location.


Binary Packages
===============

Anaconda
~~~~~~~~

It is recommended to use the binary **Anaconda** Python distribution, as it is
easy to install the difficult to build packages NumPy and SciPy under MacOSX
and Windows.

Get it from http://docs.continuum.io/anaconda/

After setting it up, you can install Emma via the conda package manager.

::

   conda install emma

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


