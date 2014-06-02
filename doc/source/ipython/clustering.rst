
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
    import emma2.coordinates.clustering as cluster
    import emma2.msm.io as msmio
    import emma2.msm.estimation as msmest
    import emma2.msm.analysis as msmanal
    import emma2.util.pystallone as stallone

.. parsed-literal::

    2014-02-02 14:24:56,684 emma2.util.pystallone DEBUG    init with options: "['-Xms64m', '-Xmx2000m', '-Djava.class.path=/Users/noe/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.6-x86_64.egg/emma2/util/../../lib/stallone/stallone-1.0-SNAPSHOT-jar-with-dependencies.jar/']"
    2014-02-02 14:24:56,696 emma2.util.pystallone DEBUG    default vm path: /System/Library/Frameworks/JavaVM.framework/JavaVM


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


k-means clustering
------------------

.. code:: python

    traj = coorio.read_traj("./resources/Trypsin_pc12.dat")
.. code:: python

    # cluster from file
    kmeans = cluster.kmeans("./resources/Trypsin_pc12.dat", 20, maxiter=5)
.. code:: python

    figure(figsize=(10,7))
    plot(traj[:,0],traj[:,1],marker=".")
    centers = kmeans.clustercenters()
    plot(centers[:,0],centers[:,1],marker="o",linewidth=0,color='red')



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x11b3bb490>]




.. image:: clustering_files/clustering_4_1.png


regspace clustering
-------------------

.. code:: python

    # cluster from file
    regspace = cluster.regspace("./resources/Trypsin_pc12.dat", 4.0)
.. code:: python

    figure(figsize=(10,7))
    plot(traj[:,0],traj[:,1],marker=".")
    centers = regspace.clustercenters()
    plot(centers[:,0],centers[:,1],marker="o",linewidth=0,color='red')



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x11baf79d0>]




.. image:: clustering_files/clustering_7_1.png


Cluster Assignment
------------------

.. code:: python

    dtraj = cluster.assign("./resources/Trypsin_pc12.dat", regspace, "./resources/Trypsin_pc12.dtraj")
.. code:: python

    plot(range(len(dtraj)),dtraj)



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x11be70550>]




.. image:: clustering_files/clustering_10_1.png



