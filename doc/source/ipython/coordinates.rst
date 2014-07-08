
Tutorial: Coordinate IO and transformation
==========================================

Blabla

:math:`x = \int_y f(y) dy`

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
    import emma2.msm.analysis as msmanal
    import emma2.util.pystallone as stallone

.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. parsed-literal::

    WARNING: pylab import has clobbered these variables: ['mean']
    `%pylab --no-import-all` prevents importing * from pylab and numpy


Reading and writing coordinates
-------------------------------

Supports reading from dcds and xtcs, as well as writing to dcds

.. code:: python

    # Read Coordinates and print first frame
    reader = coorio.reader("./resources/Trypsin_Ca_dt1ns.dcd")
    print reader.size()
    X0 = reader.get(0)
    print shape(X0)
    print X0[0:4]

.. parsed-literal::

    1000
    (223, 3)
    [[  2.14200139  15.05400085  13.5700016 ]
     [  4.11900043  18.71600342  15.07800007]
     [  6.10699749  21.63800049  12.51100159]
     [  3.07000113  19.77000046   9.00699806]]


.. code:: python

    # load all
    trajCA = reader.load()
    print shape(trajCA)

.. parsed-literal::

    (1000, 223, 3)


.. code:: python

    print trajCA[0,0:4]

.. parsed-literal::

    [[  2.14200139  15.05400085  13.5700016 ]
     [  4.11900043  18.71600342  15.07800007]
     [  6.10699749  21.63800049  12.51100159]
     [  3.07000113  19.77000046   9.00699806]]


.. code:: python

    # check performance
    %timeit reader.load()

.. parsed-literal::

    1 loops, best of 3: 357 ms per loop


.. code:: python

    # Load an atom selection
    sel = range(0,10)
    trajSome = reader.load(select=sel)
    print shape(trajSome)

.. parsed-literal::

    (1000, 10, 3)


.. code:: python

    # Load a frame selection
    sel = range(0,10)
    trajSparse = reader.load(frames=sel)
    print shape(trajSparse)

.. parsed-literal::

    (10, 223, 3)


.. code:: python

    # slice
    trajCA_10ns = trajCA[::10]
    np.shape(trajCA_10ns)



.. parsed-literal::

    (100, 223, 3)



.. code:: python

    # Write dcd
    coorio.write_traj("./resources/Trypsin_Ca_dt10ns.dcd",trajCA_10ns)
    # this has the same effect
    writer = coorio.writer("./resources/Trypsin_Ca_dt10ns.dcd",nframes=100,natoms=669)
    writer.addAll(trajCA_10ns)
    writer.close()
Reading and writing ASCII
-------------------------

ASCII coordinates are no difference. By default, tabulated ASCII files
are interpreted as a frame per line. Thus we cannot explicitly encode
coordinate tables such as Nx3. ASCII files are useful for transformed
data, such as angles, distances, principal components, etc.

In principle, python has more than enough support for reading and
writing ASCII files. The only reason why we offer it through the coorio
package is to allow coordinate manipulation using the same interface,
irrespective of the file format.

.. code:: python

    # Read Coordinates from a tabulated ASCII file
    reader = coorio.reader("./resources/dists.dat")
    print shape(reader.get(0))
    # load two coordinates
    trajXY = reader.load(select=[20,21])
    print np.shape(trajXY)

.. parsed-literal::

    (50,)
    (10000, 2)


.. code:: python

    # Write Coordinates to a tabulated ASCII file
    writer = coorio.writer("./resources/dists_2021.dat")
    writer.addAll(reader.load(select=[20,21]))
Note: data transfer across the python/Java interface is currently rather
slow. This slowdown arises currently on the python side, where the
operation '[:]' on JPype Arrays causes the slowdown. Perhaps this can be
circumvented somehow

Example: Converting dcd files to inner coordinates
--------------------------------------------------


.. code:: python

    # we define a coordinate transform
    Tdist = coortrans.createtransform_distances(range(0,10),range(0,4))
    print Tdist.dimension()

.. parsed-literal::

    40


.. code:: python

    Tdist.transform(trajCA[0])



.. parsed-literal::

    array([[  0.        ,   4.42638084,   7.7583327 ,   6.62743065],
           [  4.42638084,   0.        ,   4.36802935,   6.25046996],
           [  7.7583327 ,   4.36802935,   0.        ,   4.99908109],
           [  6.62743065,   6.25046996,   4.99908109,   0.        ],
           [  9.4730669 ,   8.46684791,   7.2295973 ,   4.74989563],
           [  9.94134796,  10.46378398,  10.8614857 ,   7.61380931],
           [ 11.28466722,  12.84710488,  12.90407465,   8.39772646],
           [ 15.12288536,  16.08321504,  16.21434571,  12.55268884],
           [ 17.81510191,  19.43892995,  19.91524109,  15.90903928],
           [ 18.61893415,  21.0618861 ,  21.89735747,  17.45178493]])



.. code:: python

    # and entire files
    infile = "./resources/Trypsin_Ca_dt1ns.dcd"
    outfile = "./resources/tmp.dat"
    # apply transformation Tdist to input file and write result to output file
    coortrans.transform_file(infile, Tdist, outfile, output_precision=(3,2))
    # check result
    print np.reshape(np.loadtxt(outfile)[0],(10,4))

.. parsed-literal::

    [[  0.     4.43   7.76   6.63]
     [  4.43   0.     4.37   6.25]
     [  7.76   4.37   0.     5.  ]
     [  6.63   6.25   5.     0.  ]
     [  9.47   8.47   7.23   4.75]
     [  9.94  10.46  10.86   7.61]
     [ 11.28  12.85  12.9    8.4 ]
     [ 15.12  16.08  16.21  12.55]
     [ 17.82  19.44  19.92  15.91]
     [ 18.62  21.06  21.9   17.45]]


.. code:: python

    # angular transforms
    Tangle = coortrans.createtransform_angles([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
    Tangle.transform(trajCA[0])



.. parsed-literal::

    array([  83.42304353,   95.69762282,  110.59752966,  101.70707066])



.. code:: python

    # dihedral transforms
    Tdih = coortrans.createtransform_dihedrals([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
    Tdih.transform(trajCA[0])



.. parsed-literal::

    array([  98.14959127, -133.14783449,  -63.41224959, -129.11232329])



.. code:: python

    # select all torsion angles along the backbone
    nCA = np.shape(trajCA)[1]
    sel_all_CA_dih = np.array([range(0,nCA-3),range(1,nCA-2),range(2,nCA-1),range(3,nCA)]).T
    Tdih_all = coortrans.createtransform_dihedrals(sel_all_CA_dih)
    print sel_all_CA_dih[0:5]
    print np.shape(sel_all_CA_dih)

.. parsed-literal::

    [[0 1 2 3]
     [1 2 3 4]
     [2 3 4 5]
     [3 4 5 6]
     [4 5 6 7]]
    (220, 4)


.. code:: python

    # compute all of them in memory
    coortrans.transform_trajectory(trajCA, Tdih_all)



.. parsed-literal::

    array([[  23.7407777 ,   98.14959127, -133.14783449, ...,   34.07858563,
              50.11135759,   40.95220397],
           [  16.77464408,  108.04372915, -133.07509846, ...,   37.87805828,
              36.81456897,   50.77995268],
           [   0.84647355,  127.02355369, -124.06285148, ...,   41.27550278,
              46.09259913,   41.44081048],
           ..., 
           [   9.50800587,   99.00293579, -127.25956962, ...,   35.68802949,
              81.9393266 ,   -3.17814515],
           [  21.15672999,  107.7474879 , -139.95899332, ...,   41.88499296,
              74.87716638,    9.61164   ],
           [  17.79014494,  100.08219122, -119.21720831, ...,   73.44323794,
             143.40233089,   16.6579599 ]])



.. code:: python

    # let's try the same on a file:
    infile = "./resources/Trypsin_Ca_dt1ns.dcd"
    outfile = "./resources/tmp2.dat"
    coortrans.transform_file(infile, Tdih_all, outfile, output_precision=(3,2))
Oh, my god. Writing from file to file is much faster than doing
everything in memory but through the python/java interface. This is
embarassing. We should really clean this up!

.. code:: python

    # Finally we try the minRMSD transform
    infile = "./resources/Trypsin_Ca_dt1ns.dcd"
    outfile = "./resources/tmp3.dat"
    T_minrmsd = coortrans.createtransform_minrmsd(trajCA[0])
    coortrans.transform_file(infile, T_minrmsd, outfile, output_precision=(3,2))
.. code:: python

    rmsf = np.loadtxt(outfile)
    plot(range(len(rmsf)),rmsf)
    xlabel('MD frame')
    ylabel('root mean square fluctuation')



.. parsed-literal::

    <matplotlib.text.Text at 0x1262c5e10>




.. image:: coordinates_files/coordinates_26_1.png


That's fast. All file-based operations seem fine!

PCA
---


.. code:: python

    pca = coortrans.pca("./resources/Trypsin_Ca_dt1ns.dcd")
.. code:: python

    # Let's have a look at the eigenvalues
    ev = pca.eigenvalues()
    plot(range(1,len(ev)+1),ev,marker='o')
    xlim(0,20)
    xlabel('principal component')
    ylabel('variance')



.. parsed-literal::

    <matplotlib.text.Text at 0x1255f3cd0>




.. image:: coordinates_files/coordinates_30_1.png


.. code:: python

    # The two first dimension contribute a lot of the variance. So let's project on these two
    pca.set_dimension(2) # we could have done this in the construction of pca already: coortrans.pca(input, ndim=2)
    # This is very slow due to reading. 
    Y = np.zeros((reader.size(), 2))
    for i in range(len(Y)):
        Y[i] = pca.transform(reader.get(i).flatten())
.. code:: python

    figure(figsize=(15,6))
    # scatter plot of pc1 and 2
    subplot2grid((1,2),(0,0))
    plot(Y[:,0],Y[:,1],marker='o',linewidth=0)
    # time trace
    subplot2grid((1,2),(0,1))
    plot(range(len(Y[:,0])),Y[:,0])
    plot(range(len(Y[:,1])),Y[:,1])



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x127e2e050>]




.. image:: coordinates_files/coordinates_32_1.png


.. code:: python

    # PCA is also a transform and can be used to transform files
    coortrans.transform_file("./resources/Trypsin_Ca_dt1ns.dcd",pca,"./resources/Trypsin_pc12.dat")
TICA
----


.. code:: python

    tica = coortrans.tica("./resources/Trypsin_Ca_dt1ns.dcd")
.. code:: python

    evals = tica.eigenvalues()
    plot(range(len(evals)),evals)
    xlim(0,10)



.. parsed-literal::

    (0, 10)




.. image:: coordinates_files/coordinates_36_1.png


.. code:: python

    # The two first dimension contribute a lot of the variance. So let's project on these two
    tica.set_dimension(2) # we could have done this in the construction of pca already: coortrans.pca(input, ndim=2)
    # This is very slow due to reading. 
    Y = np.zeros((reader.size(), 2))
    for i in range(len(Y)):
        Y[i] = tica.transform(reader.get(i).flatten())
.. code:: python

    figsize(10,7)
    plot(Y[:,0],Y[:,1],marker='o',linewidth=0)



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x11b493c90>]




.. image:: coordinates_files/coordinates_38_1.png


PCA in memory
-------------

This is just a side note. Loading coordinates into memory and performing
PCA there is way faster (same with TICA). That means there's a
performance leak somewhere in the Java code.

.. code:: python

    trajCA = reader.load()
    X = np.reshape(trajCA,(1000,223*3))
.. code:: python

    mean = np.mean(X, axis=0)
    Cov = np.dot((X - mean).T,X - mean) / (1.0*len(X))
.. code:: python

    evals,evecs = np.linalg.eig(Cov)
.. code:: python

    plot(range(len(evals)),evals)
    xlim(0,10)



.. parsed-literal::

    (0, 10)




.. image:: coordinates_files/coordinates_43_1.png


.. code:: python

    Y = np.zeros((reader.size(), 2))
    for i in range(len(Y)):
        Y[i] = np.dot(trajCA[i].flatten(),evecs[:,0:2])
.. code:: python

    figsize(10,7)
    plot(Y[:,0],Y[:,1],marker='o',linewidth=0)



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x12541b650>]




.. image:: coordinates_files/coordinates_45_1.png


.. code:: python

    