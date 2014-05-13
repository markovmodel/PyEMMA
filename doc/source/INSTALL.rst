.. _ref_install:

============
Installation
============

To install the Emma2 Python package, you need a few Python package dependencies
like NumPy. With recent versions of setuptools, these will be installed automatically. 
The software is being developed and tested on Python-2.7.

Setuptools
==========
It is recommended to upgrade to latest setuptools with the provided 
script 'ez_setup.py'. Invoke it like that to upgrade:

::

    python ez_setup.py [--user]


Java/Python Bridge
==================
For the Java/Python bridge provided by **Jpype** you need to install a recent
Java Development Kit (JDK). If the setup fails during the installation of Jpype,
e.g with an error like 'No JDK found.', please have look at the install notes here
http://jpype.readthedocs.org/en/latest/install.html

Install 
On Debian/Ubuntu:

::

    apt-get install openjdk-7-jdk
    

On MacOs obtain a copy from Oracle:
http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html

Please have a look at :ref:`ref-install-mac`.


Dependencies
============
These Python packages are needed, and will be installed automatically, when you
invoke the setup.py script (no matter if via easy_install, pip or directly).

Python packages:

- numpy >= 1.6.0
- scipy >= 0.13.0
- jpype1 >= 0.5.5.1


If you intend to build the Sphinx documentation yourself, you also need the
following additional Python packages:

- sphinx >= 1.2.1
- numpydoc >= 0.4

.. _ref-install-methods:

Install methods
===============
For all install methods you can append '--user' to install to your local user
directory assuming you are in the root of the repository:

- Install with setup.py
 ::

    python setup.py install [--user]

- Install with pip package manager
 ::

    pip install . [--user]

- Install with easy_install package manager
 ::

    easy_install [--user] .

Note that the dot in invocation of pip and easy_install are necessary to point
to the current dir, where setup.py is located.
