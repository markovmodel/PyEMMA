import jsonpickle

from pyemma._base.logging import Loggable
from pyemma.util.types import is_string
import contextlib


class LoadedObjectVersionMismatchException(Exception):
    """ the version does not match the current version of the library. """


class DeveloperError(Exception):
    """ the devs has done something wrong. """


def load(file_like):
    import bz2
    if is_string(file_like):
        file_like = bz2.BZ2File(file_like)
    with contextlib.closing(file_like) as file_like:
        inp = file_like.read()
        obj = jsonpickle.loads(inp)

    return obj


class SerializableMixIn(object):
    """

    """

    def save(self, filename_or_filelike):
        """
        Parameters
        -----------
        filename_or_filelike: str or file like
            path to desired output file or a type which implements the file protocol.
        """
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            if isinstance(self, Loggable):
                self.logger.exception('During saving the object ("%s")'
                                      'the following error occured' % e)
            raise

        flattened = bytes(flattened)

        import bz2
        with contextlib.closing(bz2.BZ2File(filename_or_filelike, 'w')) as fh:
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
