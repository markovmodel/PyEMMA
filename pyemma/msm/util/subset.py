# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as _np


class SubSet(object):
    @property
    def active_set(self):
        if not hasattr(self, "_active_set"):
            self._active_set = []
        return self._active_set

    @active_set.setter
    def active_set(self, active_set):
        if active_set is None:
            active_set = []
        self._active_set = _np.asarray(active_set, dtype=_np.intc)

    @property
    def nstates_full(self):
        if not hasattr(self, "_nstates_full"):
            self._nstates_full = len(self.active_set)
        return self._nstates_full

    @nstates_full.setter
    def nstates_full(self, nstates_full):
        try:
            self._nstates_full = int(nstates_full)
        except TypeError:
            self._nstates_full = None


def _globalize(data, axis, active_set, default_value, n_centers):
    if data.ndim == 1:
        array = _np.asarray([default_value]).repeat(n_centers)
        array[active_set] = data
    elif data.ndim == 2:
        expanded_shape = list(data.shape)
        expanded_shape[axis] = n_centers
        array = _np.asarray([default_value]).repeat(
            expanded_shape[0] * expanded_shape[1]).reshape(expanded_shape)
        if axis == 0:
            array[active_set, :] = data
        elif axis == 1:
            array[:, active_set] = data
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return array


def _wrap_to_full_state(name, default_value, axis):
    from pyemma._base.estimator import _call_member

    def alias_to_full_state(self, *args, **kw):
        data = _call_member(self, name, *args, **kw)
        data = _np.asarray(data)
        return _globalize(data, axis, self.active_set, default_value, self.nstates_full)

    return alias_to_full_state


def add_full_state_methods(class_with_globalize_methods):
    """
    class decorator to create "_full_state" methods/properties on the class (so they
    are valid for all instances created from this class).

    Parameters
    ----------
    class_with_globalize_methods

    """
    assert hasattr(class_with_globalize_methods, 'active_set')
    assert hasattr(class_with_globalize_methods, 'nstates_full')

    for name, method in class_with_globalize_methods.__dict__.copy().items():
        if isinstance(method, property) and hasattr(method.fget, '_map_to_full_state_def_arg'):
            default_value = method.fget._map_to_full_state_def_arg
            axis = method.fget._map_to_full_state_along_axis
            new_getter = _wrap_to_full_state(name, default_value, axis)
            alias_to_full_state_inst = property(new_getter)
        elif hasattr(method, '_map_to_full_state_def_arg'):
            default_value = method._map_to_full_state_def_arg
            axis = method._map_to_full_state_along_axis
            alias_to_full_state_inst = _wrap_to_full_state(name, default_value, axis)
        else:
            continue

        name += "_full_state"
        setattr(class_with_globalize_methods, name, alias_to_full_state_inst)

    return class_with_globalize_methods


class map_to_full_state(object):
    """ adds a copy of decorated method/property to be passed to the full state interpolation function

    Parameters
    ----------
    default_arg: object
        the default argument to interpolate missing values with.
    extend_along_axis : int, default=0
        extend along given axis for multi-dimensional data.
    """

    def __init__(self, default_arg, extend_along_axis=0):
        self.default_arg = default_arg
        self.extend_along_axis = extend_along_axis

    def __call__(self, func):
        if isinstance(func, property):
            func.fget._map_to_full_state_def_arg = self.default_arg
            func.fget._map_to_full_state_along_axis = self.extend_along_axis
        else:
            func._map_to_full_state_def_arg = self.default_arg
            func._map_to_full_state_along_axis = self.extend_along_axis
        return func
