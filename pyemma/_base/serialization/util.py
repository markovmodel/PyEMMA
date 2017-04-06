
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
        self.locations = locations

    def __call__(self, *args, **kwargs):
        from .serialization import _renamed_classes
        locations = self.locations
        assert all(isinstance(x, str) for x in locations)
        new_class = args[0]
        for old in locations:
            _renamed_classes[old] = new_class
