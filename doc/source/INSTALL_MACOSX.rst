.. _ref-install-mac:

=====================
Install under Mac OSX
=====================

If you simply want to use Emma it is absolutly fine to go for binary packages.

Install binary Packages
=======================
It is recommended to use the pre-built **Anaconda** Python distribution.
Get it from http://docs.continuum.io/anaconda/

It already contains most of Emma's dependencies like *NumPy* and *SciPy*, which are
not easy to build under MacOS.

.. TODO: If we have binary packages for mac update this guide.

Never the less you need an C/C++ compiler to build binary extensions. If you 
intend to build **Scipy** from source, you will also need a Fortran compiler.

If you have Anaconda up and running, you install Emma via:

::

    pip install emma2 [--user]
    
    
The *--user* argument is optional and tells pip to install into your home folder,
so you do not need root permissions.

You get binary distributions of these in dmg packages, which can be easily 
installed. Note that you may need admin privilegdes to do so.

2. install python-2.7 from python.org; run "Update shell Profile.command" to 
   update your PATH to point to the installation of new python
3. If you set a JAVA_HOME environment variable to point to your JDK and use

::
    sudo to install jpype

(inclusive in step 5), make sure to use sudo -E to preserve the JAVA_HOME variable!

4. install numpy (latest) from http://numpy.org
5. install scipy (latest) from http://scipy.org 
   
Building from Source
====================
If you are a developer or want to have optimized builds for your platform, you
may want to build it from source. This also involves building scipy and numpy
from source.

Prequisites
^^^^^^^^^^^
 * C/C++ compiler
 * Fortran compiler

You may find this .. _information: http://www.scipy.org/scipylib/building/macosx.htm 
useful to obtain the neccessary building tools.

Building/Installing
^^^^^^^^^^^^^^^^^^^
The build and install process is in one step, as all dependencies are dragged in 
via the provided *setup.py* script. So you only need to get the source of Emma 
and run it to build Emma itself and all of its dependencies from source.

1. Obtain a clone via

::

   git clone https://github.com/cmb-fu/emma.git

2. install emma2 via

::

  python setup.py install [--user]
  
or via a method described here :ref:`ref-install-methods`.
