from pyemma._ext import jsonpickle

from pyemma._base.logging import Loggable
from pyemma.util.types import is_string
import contextlib


class LoadedObjectVersionMismatchException(Exception):
    """ the version does not match the current version of the library. """


class DeveloperError(Exception):
    """ the devs has done something wrong. """


def load(file_like):
    import bz2

    from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler
    register_ndarray_handler()
    if is_string(file_like):
        file_like = bz2.BZ2File(file_like)
    with contextlib.closing(file_like) as file_like:
        inp = file_like.read()
        inp = str(inp, encoding='ascii')
        obj = jsonpickle.loads(inp)

    return obj


class SerializableMixIn(object):
    """

    """

    def save(self, filename):
        """
        Parameters
        -----------
        filename: str or file like
            path to desired output file or a type which implements the file protocol.
        """
        from pyemma._base.serialization.jsonpickler_handlers import register_ndarray_handler
        register_ndarray_handler()
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            if isinstance(self, Loggable):
                self.logger.exception('During saving the object ("%s")'
                                      'the following error occured' % e)
            raise

        flattened = bytes(flattened, encoding='ascii')

        import bz2
        with contextlib.closing(bz2.BZ2File(filename, 'w')) as fh:
            try:
                fh.write(flattened)
            except:
                raise

    @classmethod
    def load(cls, file_like):
        """
        Parameter
        ---------
        file_like : str or file like object (has to provide read method).
            The file like object tried to be read for a serialized object.

        Returns
        -------
        obj : the de-serialized object
        """
        obj = load(file_like)

        if obj.__class__ != cls:
            raise ValueError("Given file '%s' did not contain the right type:"
                             " desired(%s) vs. actual(%s)" % (file_like, cls, obj.__class__))
        if not hasattr(cls, '_version'):
            pass
            #raise DeveloperError("your class does not implement the deserialization protocol of PyEMMA.")

        if obj._version != cls._version:
            pass
            #raise LoadedObjectVersionMismatchException("Version mismatch")

        return obj
