devel Package
=============

:mod:`function_template` Module
-------------------------------

.. automodule:: devel.function_template
    :members:
    :undoc-members:
    :show-inheritance:


Additional comments
-------------------

The part above was automatically generated from the docstring of the following function:

.. code-block:: python

   from numpy import *

   def check_nonnegativity(A):
       """Checks nonnegativity of a matrix.

       Matrix A=(:math:`a_{ij}`) is nonnegative if
       :math:`a_{ij} \geq 0` for all :math:`i, j`.

       Parameters
       ----------
       A : ndarray, shape=(M, N)
           The matrix to test.

       Returns
       -------
       nonnegative : bool
           The truth value of the nonnegativity test.

       Notes
       -----
       The nonnegativity test is performed using
       boolean ndarrays.

       Nonnegativity is import for transition matrix estimation.

       Examples
       --------
       >>> import numpy as np
       >>> A=np.array([[0.4, 0.1, 0.4], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
       >>> x=check_nonnegativity(A)
       >>> x
       True

       >>> B=np.array([[1.0, 0.0], [2.0, 3.0]])
       >>> x=check_nonnegativity(A)
       >>> x
       False

       """
       ind=(A<=0.0)
       return sum(ind)==0

To autogenerate the documentation for a module, follow this scheme:

.. code-block:: bash

   cd ~/git/emma2/                   # this is my local emma repository
   cd sphinx                         # and this is the documentation directory

   sphinx-apidoc -o . ../devel       # this creates the documentation files
                                     # for the module ../devel in the
                                     # sphinx directory

   make html                         # run the html builder to create the
                                     # html documentation from the .rst sources

   firefox _build/html/index.html    # see what you have done

