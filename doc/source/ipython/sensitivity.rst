
emma2 MSM-analysis for alanine-dipeptide
========================================

This notebook collects usage examples for MSM-analysis using members of
the emma2.msm.analysis package.

A given MSM, estimated from alanine-dipeptide simulation data at lagtime
:math:`\tau=6ps`, is used as an example to carry out analysis.

The necessary inputs are:

1. the transition matrix, 'T.dat'
2. the centers of the :math:`(\phi, \psi)` dihedral angle space regular
   grid discretization, 'grid\_centers20x20.dat'
3. the largest set of connected microstates, 'lcc.dat'

Auxiliary functions in 'plotting.py' are used to generate figures of the
estimated quantities.

Use ipythons magic % commands to activate plotting within notebook cells
------------------------------------------------------------------------


.. code:: python

    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'
Imports are ordered as
----------------------

1. standard library imports
2. third party imports
3. local application/library specific imports


.. code:: python

    import matplotlib.pyplot as plt
.. code:: python

    import numpy as np
.. code:: python

    from emma2.msm.io import read_matrix

.. parsed-literal::

    2014-02-05 16:47:07,549 emma2.util.pystallone DEBUG    init with options: "['-Xms64m', '-Xmx2000m', '-Djava.class.path=/Users/jan-hendrikprinz/anaconda/lib/python2.7/site-packages/Emma2-2.0-py2.7-macosx-10.5-x86_64.egg/emma2/util/../../lib/stallone/stallone-1.0-SNAPSHOT-jar-with-dependencies.jar/']"
    2014-02-05 16:47:07,552 emma2.util.pystallone DEBUG    default vm path: /System/Library/Frameworks/JavaVM.framework/JavaVM


.. code:: python

    from emma2.msm.analysis import stationary_distribution, eigenvectors, eigenvalues, timescales, pcca
.. code:: python

    import plotting
Load necessary input data
-------------------------

Use emma2.msm.io.read\_matrix function to read dense arrays from ascii
files. The returned objects will be dense arrays (numpy.ndarray).

.. code:: python

    T=read_matrix('T.dat')
This notebook collects usage examples for MSM-analysis using members of
the emma2.msm.analysis package.

Starting from

.. code:: python

    centers=read_matrix('grid_centers20x20.dat')
The optional dtype (data-type) keyword allows you to specify the type of
the read data. The default value is dtype=float.

.. code:: python

    lcc=read_matrix('lcc.dat', dtype=int)
Use the integer values given by the largest connected set as indices to
"slice" the array of grid-center points. The returned array contains
only those centers corresponding to the mircrostates in the largest
connected set.

.. code:: python

    centers=centers[lcc, :]
Compute the stationary distribution using the
emma2.msm.analysis.stationary\_distribution method.

.. code:: python

    traj = 
Stationary Distribution
-----------------------


.. code:: python

    pi=stationary_distribution(T)
The (centers, pi) tuple is fed into an adapted plotting subroutine
producing a contour plot from the scattered data. Since scatterd data
can not directly be used to produce a contour plot over the whole
:math:`(\phi, \psi)`-plane the given data is interpolated onto a regular
grid before producing a contour plot. Some of the strange-looking low
probability iso-lines may be artefacts of the interpolation.
Interpolation on the level of free energies is probably a better idea.

.. code:: python

    plotting.stationary_distribution(centers, pi)


.. image:: sensitivity_files/sensitivity_22_0.png


Eigenvectors
------------


We compute the right eigenvectors corresponding to the 4 largest
eigenvalues.

.. code:: python

    R=eigenvectors(T, k=4)
The first eigenvector shows a sign change from the most stable region
with :math:`\phi \leq 0` to the :math:`\phi>0` region. The slowest
process corresponds to a transition between the two most stable states
and the metastable regions with :math:`\phi>0`.

.. code:: python

    ev=R[:, 1].real
    plotting.eigenvector(centers, ev, levels=np.linspace(ev.min(), ev.max(), 10))


.. image:: sensitivity_files/sensitivity_27_0.png


The second eigenvector shows a sign change from :math:`\phi \leq 0` to
:math:`\phi>0`. The second slowest process is the transition between the
low-probability region :math:`\phi>0` and the high probability region
:math:`\phi \leq 0`.

.. code:: python

    ev=R[:, 2].real
    plotting.eigenvector(centers, ev, levels=np.linspace(ev.min(), ev.max(), 11), fmt='%.e')


.. image:: sensitivity_files/sensitivity_29_0.png


The third eigenvector shows the transition process between the least
probable meta-stable state and the rest of the accessible state space.

.. code:: python

    ev=R[:, 1].real
    plotting.eigenvector(centers, ev, levels=np.linspace(ev.min(), ev.max(), 21), fmt='%.e')


.. image:: sensitivity_files/sensitivity_31_0.png


Eigenvalues
-----------

Compute the 10 largest eigenvalues of the MSM

.. code:: python

    eigvals=eigenvalues(T)[0:11]
The first :math:`5` eigenvalues are purely real. The remaining
eigenvalues occur in complex-conjugate pairs. That is because :math:`T`
is a matrix with purely real entries.

.. code:: python

    eigvals



.. parsed-literal::

    array([ 1.00000000+0.j        ,  0.94808553+0.j        ,
            0.94092025+0.j        ,  0.66447475+0.j        ,
            0.38530146+0.j        ,  0.34550046+0.00929879j,
            0.34550046-0.00929879j,  0.24977533+0.25204877j,
            0.24977533-0.25204877j,  0.23257796+0.19019451j,
            0.23257796-0.19019451j])



There is a distinct gap in the spectrum betwenn the third and the fourth
eigenvalue.

.. code:: python

    plotting.eigenvalues(eigvals)

.. parsed-literal::

    /Users/jan-hendrikprinz/anaconda/lib/python2.7/site-packages/numpy/core/numeric.py:320: ComplexWarning: Casting complex values to real discards the imaginary part
      return array(a, dtype, copy=False, order=order)



.. image:: sensitivity_files/sensitivity_37_1.png


Implied time scales
-------------------

Implied time scales are computed via msm.analysis.timescales. The
lagtime of the Markov model, :math:`\tau=6 ps`, can be specified via the
optional keyword tau. The default value is tau=1.

.. code:: python

    ts=timescales(T, k=5, tau=6)
.. code:: python

    ts



.. parsed-literal::

    array([          inf,  112.54805277,   98.52720209,   14.67859726,
              6.29109374])



PCCA
----


Ufortunately we seem to have a bug in the current implementation. So
that pcca(T, 5) will produce a nasty stack trace. In stead we load the
membership computed by a MATLAB script to visualize the result that
should have been produced.

.. code:: python

    membership=np.loadtxt('membership.dat')
.. code:: python

    result=pcca(T,3)
.. code:: python

    membership_crisp=np.where(result>0.50)
PCCA gives accurate memberships for the high probability region.
Assigning correct memberships for the low probability states,
:math:`\phi>0`, is problematic.

.. code:: python

    plotting.pcca(centers, membership_crisp)


.. image:: sensitivity_files/sensitivity_47_0.png


.. code:: python

    re=result[:,2]
    plotting.eigenvector(centers, re, levels=np.linspace(re.min(), re.max(), 11), fmt='%.e')


.. image:: sensitivity_files/sensitivity_48_0.png


.. code:: python

    from emma2.msm.estimation import tmatrix_cov
    from emma2.msm.estimation import error_perturbation
    from emma2.msm.analysis import stationary_distribution_sensitivity
    from emma2.msm.estimation import transition_matrix
.. code:: python

    C = np.array( [[10,2,1], [2,12,3], [1,3,2]] )*1
.. code:: python

    T = transition_matrix(C)
.. code:: python

    SDSens = stationary_distribution_sensitivity(T,0)
.. code:: python

    cov = tmatrix_cov(C)
.. code:: python

    SD = stationary_distribution(T)
    
    for i in range(0,3):
        SDSens = stationary_distribution_sensitivity(T,i)
        print SD[i], "+/-", np.sqrt(error_perturbation(C, SDSens))

.. parsed-literal::

    0.361111111111 +/- 0.164191464335
    0.472222222222 +/- 0.14402754825
    0.166666666667 +/- 0.0757566293387


Summary
-------

The emma2.msm.analysis module can be used to analyse an estimated
transition matrix. Starting from the transition matrix :math:`T` It is
possible to

-  compute the stationary vector :math:`\pi` to analyze the free-energy
   landscape given suitable (low-dimensional) coordinates
-  compute the right eigenvectors to investigate slowest dynamical
   processes
-  compute eigenvalues and time scales as quantitative information about
   system-dynamics

The emma2.msm-API is designed to allow fast and flexible scripting of
the whole estimation and analysis process. There is a multitude of
functions for MSM analysis provided in the emma2.msm.analysis module.
Further functions are

-  checks for stochasticity, ergodicity, etc.
-  commitor computation
-  TPT
-  mean-first-passage time (mfpt) computations
-  fingerprint: expectation and autocorrelation
-  decompositions in eigenvalues, left, and right eigenvectors

We are happy for your feedback and suggestions. Please feel free to
contact our mailing list at emma@lists.fu-berlin.de

.. code:: python

    