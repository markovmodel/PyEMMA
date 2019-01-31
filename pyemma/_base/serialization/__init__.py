

"""

.. module:: pyemma


Model serialization
===================

In PyEMMA most Estimators and Models can be saved to disk in an efficient file format.
Most of the Estimators and Models in PyEMMA are serializable. If a given PyEMMA object can be saved to disk,
it provides a save method.


To load a previously saved Estimator or Model there is the load function. Since a file can contain multiple
Estimators/Models saved under different names, you can inspect the files with the :func:`list_models` function,
to obtain the previously used names.

.. autofunction::
   load
   list_models

Notes
-----

We try our best to provide future compatibility of already saved data. This means it should always be possible to load
data with a newer version of the software, but you can not do reverse, eg. load a model saved by a new version with
an old version of PyEMMA.

"""


def load(filename, model_name='default'):
    """ Restores a previously saved PyEMMA object from disk.

    Parameters
    ----------
    filename : str
        path to filename, where the model has been stored.
    model_name: str, default='default'
        if multiple objects are contained in the file, these can be accessed by
        their name. Use :func:`pyemma.list_models` to get a representation of all stored models.

    Returns
    -------
    obj : Model or Estimator
        the instance containing the same parameters as the saved model/estimator.

    """
    from .serialization import SerializableMixIn
    return SerializableMixIn.load(file_name=filename, model_name=model_name)


def list_models(filename):
    """ Lists all models in given filename.

    Parameters
    ----------
    filename: str
        path to filename, where the model has been stored.

    Returns
    -------
    obj: dict
        A mapping by name and a comprehensive description like this:
        {model_name: {'repr' : 'string representation, 'created': 'human readable date', ...}
    """
    from .h5file import H5File
    with H5File(filename, mode='r') as f:
        return f.models_descriptive

# TODO: how to pretend these functions are living top package level?
#load.__module__  = list_models.__module__= 'pyemma'
