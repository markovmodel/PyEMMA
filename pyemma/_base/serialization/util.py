from pyemma._ext.jsonpickle.util import importable_name


class _old_locations(object):
    """ Updates the renamed classes dictionary for serialization handling.

    The idea is to provide a location/name history to the current decorated class,
    so old occurrences can be easily mapped to the current name upon loading old models.

    Parameters
    ----------
    locations: list, tuple of string
        the elements are dotted python names to classes.
    """

    def __init__(self, locations):
        assert all(isinstance(x, (str, bytes)) for x in locations)
        import re
        # restrict to pyemma and valid module names: https://regex101.com/r/vjcWT8/1
        # FIXME: this does not allow underscores in names...
        # assert all(re.fullmatch("pyemma(\.[a-zA-Z]+[0-9]*)+", s) for s in locations)
        #locations = map(bytes, locations)
        self.locations = locations

    def __call__(self, *args, **kwargs):
        from .serialization import _renamed_classes
        cls = args[0]
        new_class = importable_name(cls)

        # import six
        #new_name = "{module}.{cls}".format(module=cls.__module__, cls=cls.__name__)
        #kw = {} if six.PY2 else {'encoding': 'ascii'}
        #new_class = bytes(new_name, **kw)

        for old in self.locations:
            _renamed_classes[old] = new_class

        return cls
