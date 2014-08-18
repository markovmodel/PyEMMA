
.. code:: python

    ##########################################
    # IMPORT ALL REQUIRED PACKAGES
    ##########################################
    # system
    import os
    import math
    # numerics 
    import numpy as np
    import scipy.sparse as sparse
    from scipy.sparse.base import issparse
    # iPython 
    from IPython.display import display
    # matplotlib
    import matplotlib.pyplot as plt
    %pylab inline
    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    #emma imports
    import emma2.coordinates.io as coorio
    import emma2.coordinates.transform as coortrans
    import emma2.msm.io as msmio
    import emma2.msm.estimation as msmest
    import emma2.msm.analysis as msmana
    import emma2.util.pystallone as stallone

::


    ---------------------------------------------------------------------------
    ImportError                               Traceback (most recent call last)

    <ipython-input-2-4f65cf2b7803> in <module>()
         17 from mpl_toolkits.mplot3d import Axes3D
         18 #emma imports
    ---> 19 import emma2.coordinates.io as coorio
         20 import emma2.coordinates.transform as coortrans
         21 import emma2.msm.io as msmio


    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/__init__.py in <module>()
          4 
          5 import coordinates
    ----> 6 import msm
          7 import pmm
          8 import util


    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/msm/__init__.py in <module>()
         10 
         11 import analysis
    ---> 12 import estimation
         13 import generation
         14 import io


    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/msm/estimation/__init__.py in <module>()
          3 """
          4 
    ----> 5 from api import *
    

    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/msm/estimation/api.py in <module>()
         14 import sparse.likelihood
         15 import sparse.transition_matrix
    ---> 16 import sparse.perturbation
         17 import dense.transition_matrix
         18 


    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/msm/estimation/sparse/perturbation.py in <module>()
          4 @author: jan-hendrikprinz
          5 '''
    ----> 6 from .. import api as msmest
          7 
          8 import numpy


    ImportError: cannot import name api


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    C = np.array([[7,4,0],[1,1,1],[1,2,5]])
.. code:: python

    T = msmest.tmatrix(C)
.. code:: python

    Sstat2 = msmana.stationary_distribution_sensitivity(T,2)
.. code:: python

    msmest.tmatrix_cov(1.0*C)



.. parsed-literal::

    array([[[ 0.01928375, -0.01928375,  0.        ],
            [-0.01928375,  0.01928375,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],
    
           [[ 0.05555556, -0.02777778, -0.02777778],
            [-0.02777778,  0.05555556, -0.02777778],
            [-0.02777778, -0.02777778,  0.05555556]],
    
           [[ 0.01215278, -0.00347222, -0.00868056],
            [-0.00347222,  0.02083333, -0.01736111],
            [-0.00868056, -0.01736111,  0.02604167]]])



.. code:: python

    msmest.error_perturbation(C, Sstat2)

::


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-11-b7f24ade9897> in <module>()
    ----> 1 msmest.error_perturbation(C, Sstat2)
    

    /Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/msm/estimation/api.pyc in error_perturbation(C, sensitivity)
        343 
        344     """
    --> 345     return sparse.perturbation.error_perturbation(C, sensitivity)
        346 
        347 # DONE: Martin Map to Stallone (Reversible)


    AttributeError: 'module' object has no attribute 'perturbation'


.. code:: python

    