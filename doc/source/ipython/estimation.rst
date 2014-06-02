
Emma2 - Estimation of a Markov model for Alanine-Dipeptide (already discretized trajectory)
===========================================================================================

.. code:: python

    from emma2.msm.io import read_dtraj
    from emma2.msm.estimation import cmatrix, connected_cmatrix, tmatrix
    import matplotlib.pyplot as plt
Read discretized trajectory, :math:`dt=1ps`, :math:`T = 1000ns`
---------------------------------------------------------------

.. code:: python

    dtraj = read_dtraj('data/dihedral_dt1ps_T1000ns_1.disctraj')
.. code:: python

    dtraj
Count matrix at lagtime :math:`\tau = 6 ps`
-------------------------------------------

.. code:: python

    C_slide = cmatrix(dtraj, 6, sliding = True)
    C_slide2 = cmatrix(dtraj, 60, sliding = True)
    C = cmatrix(dtraj, 6, sliding = False)
    C60 = cmatrix(dtraj, 60, sliding = False)
Efficient construction and storage using sparse coordinate list (COO) format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    C, C60, C_slide, C_slide2
Count matrix of largest connected component
-------------------------------------------

.. code:: python

    C_cc = connected_cmatrix(C)
Extremely fast computation of connected component using scipy.csgraph library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    C_cc
Maximum likelihood transition matrix
------------------------------------

support for unconstrained (nonreversible) as well as constrained (reversible) maximum likelihood optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    Trev = tmatrix(C_cc, reversible = True)
    T = tmatrix(C_cc)
.. code:: python

    T, Trev