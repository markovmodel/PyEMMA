
class _old_locations(object):
    """ updates the renamed classes dictionary for serialization handling.

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


def dotted_name_to_json_str(dotted_name):
    pass