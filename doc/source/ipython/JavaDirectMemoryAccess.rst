
Pass memory to Java interface without copying the data around.
==============================================================

.. code:: python

    import emma2
    import numpy as np
create an empty numpy vector and show its flags. Its important that it
is *C-contiguous*, so it can be accessed in an c array like fashion (raw
pointer access).

.. code:: python

    x = np.zeros(3)
    print x.flags

.. parsed-literal::

      C_CONTIGUOUS : True
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      UPDATEIFCOPY : False


Now one can convert this numpy vector to a direct buffer the stallone
array wrapping interface. The stallone array wrapper directly converts
to direct buffer and wraps this in an stallone array type. Ensure that
you pass copy=False to the method. Otherwise a copy will be performed.
The wrapper method will also raise, if the passed in array is not
contiguous.

.. code:: python

    s = emma2.util.pystallone.ndarray_to_stallone_array(x, copy=False)
.. code:: python

    s.set(0, 24.)
    s.set(2, 42.)
.. code:: python

    print s

.. parsed-literal::

    24.0	0.0	42.0	


.. code:: python

    np.allclose(x[0], s.get(0))



.. parsed-literal::

    True



.. code:: python

    np.allclose(x, s.getArray())



.. parsed-literal::

    True


