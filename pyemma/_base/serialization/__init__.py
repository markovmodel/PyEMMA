"""
Interfacing functions and classing to Serialization.

Derive from SerializableMixIn to make your class serializable. If you need to patch an old version of your class,
you need the Modifications class.

"""

from .serialization import SerializableMixIn, Modifications

__all__ = ['SerializableMixIn', 'Modifications', 'load', 'list_models']


def load(filename, model_name='default'):
    from .serialization import SerializableMixIn
    return SerializableMixIn.load(file_name=filename, model_name=model_name)


load.__doc__ = SerializableMixIn.load.__doc__


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
