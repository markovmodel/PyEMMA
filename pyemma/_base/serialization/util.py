
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from collections import defaultdict

__author_ = 'marscher'


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
        for old in location:
            self._old_to_new[old] = new_cls
            self._new_to_old[new_cls].append(old)

    def clear(self):
        self._old_to_new.clear()
        self._new_to_old.clear()

    def find_replacement_for_old(self, old):
        return self._old_to_new.get(old, None)

    def old_handled_by(self, klass):
        return self._new_to_old.get(klass, ())


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


def _importable_name(cls):
    name = cls.__name__
    module = cls.__module__
    return '%s.%s' % (module, name)
