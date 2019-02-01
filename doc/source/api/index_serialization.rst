
Model serialization
===================

In PyEMMA most Estimators and Models can be saved to disk in an efficient file format.
Most of the Estimators and Models in PyEMMA are serializable. If a given PyEMMA object can be saved to disk,
it provides a save method:

.. automethod::
   pyemma._base.serialization.SerializableMixIn.save


Use the load function to load a previously saved PyEMMA object. Since a file can contain multiple
objects saved under different names, you can inspect the files with the :func:`pyemma.list_models` function
to obtain the previously used names. There is also a command line utility `pyemma_list_models` to inspect these files
quickly, without the need launching your own Python script.

.. autofunction::
   pyemma.load

.. autofunction::
   pyemma.list_models


Notes
-----

We try our best to provide future compatibility for previously saved data. This means it should always be possible to load
data with a newer version of the software. However, you can not do reverse; e.g., load a model saved by a new version with
an old version of PyEMMA.

