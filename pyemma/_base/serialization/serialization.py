import jsonpickle
from pyemma.util.types import is_string


def load(file_like):
    import bz2
    try:
        if is_string(file_like):
            file_like = bz2.open(file_like)

        inp = str(file_like.read(), encoding='ascii')
        obj = jsonpickle.loads(inp)

    finally:
        file_like.close()

    return obj


class SerializableMixIn(object):
    """

    """

    def save(self, filename):
        """
        Parameters
        -----------
        filename: str
            path to desired output file.
        """
        try:
            flattened = jsonpickle.dumps(self)
        except Exception as e:
            raise
        import bz2
        try:
            fh = bz2.BZ2File(filename, 'w')
            fh.write(bytes(flattened, encoding='ascii'))
        except:
            raise
        finally:
            fh.close()

    @classmethod
    def load(cls, file_like):
        """
        Parameter
        ---------
        file_like : str or file like object (has to provide read method).
            The file like object tried to be read for a serialized object.

        Returns
        -------
        obj : the serialized object
        """
        obj = load(file_like)

        if obj.__class__ != cls:
            raise ValueError("Given file '%s' did not contain the right type:"
                             " desired(%s) vs. actual(%s)" % (file_like, cls, obj.__class__))

        return obj
