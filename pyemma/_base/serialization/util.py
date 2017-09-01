from collections import defaultdict

from pyemma._ext.jsonpickle.util import importable_name


class _ClassRenameRegistry(object):
    """ perform a mapping between old and new names and reverse.
     A class can be renamed multiple times. """

    def __init__(self):
        self._old_to_new = {}
        self._new_to_old = defaultdict(list)

    def add_mapping(self, location, new_cls):
        if isinstance(location, str):
            location = [location]
        assert hasattr(new_cls, "__module__"), "makes only sense for importable classes."
        new_cls_str = importable_name(new_cls)
        for old in location:
            self._old_to_new[old] = new_cls_str
            self._new_to_old[new_cls].append(old)

    def clear(self):
        self._old_to_new.clear()
        self._new_to_old.clear()

    def find_replacement_for_old(self, old):
        return self._old_to_new.get(old, None)

    def old_handled_by(self, klass):
        return self._new_to_old.get(klass, ())

    def upgrade_old_names_in_json(self, data):

        if isinstance(data, bytes):
            data = data.decode('ascii')

        for renamed in self._old_to_new:
            new = self._old_to_new[renamed]
            data = data.replace(renamed, new)
        return data

class_rename_registry = _ClassRenameRegistry()


class handle_old_classes(object):
    """ Updates the renamed classes dictionary for serialization handling.

    The idea is to provide a location/name history to the current decorated class,
    so old occurrences can be easily mapped to the current name upon loading old models.

    Parameters
    ----------
    locations: list, tuple of string
        the elements are dotted python names to classes.
    """

    def __init__(self, locations):
        if not isinstance(locations, (tuple, list)):
            locations = [locations]
        assert all(isinstance(x, str) for x in locations)
        self.locations = locations

    def __call__(self, cls):
        class_rename_registry.add_mapping(self.locations, cls)
        return cls
