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


def print_args(func):
    def wrap(*args, **kwargs):
        print("called %s with:" % func)
        print("args:", args)
        if kwargs:
            print("kw:", kwargs)
        return func(*args, **kwargs)
    return wrap


@print_args
def globalise(data, axis, active_set, default_value, n_centers):
    shape_org = _np.shape(data)
    ndim = data.ndim
    n = n_centers if ndim == 1 else n_centers*shape_org[1]
    array = _np.asarray([default_value]).repeat(n)
    if ndim == 1:
        array[active_set] = data
    elif ndim == 2:
        array = array.reshape(-1, shape_org[1])
        if axis == 0:
            array[active_set, :] = data
        elif axis == 1:
            array[:, active_set] = data
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return array


def add_full_state_methods(class_with_globalize_methods):
    assert hasattr(class_with_globalize_methods, 'active_set')
    assert hasattr(class_with_globalize_methods, 'nstates_full')
    from .estimator import _call_member
    from types import MethodType

    def mk_f(name, default_value, axis):
        def alias_to_full_state(self, *args, **kw):
            data = _call_member(self, name, *args, **kw)
            data = _np.asarray(data)
            return globalise(data, axis, self.active_set, default_value, self.nstates_full)
        return alias_to_full_state

    original_methods = class_with_globalize_methods.__dict__.copy()
    for name, method in original_methods.iteritems():
        if not hasattr(method, '_map_to_full_state_def_arg'):
            continue

        default_value = method._map_to_full_state_def_arg
        axis = method._map_to_full_state_along_axis
        alias_to_full_state = mk_f(name, default_value, axis)

        alias_to_full_state.__doc__ = method.__doc__
        name += "_full_state"
        new_method = MethodType(alias_to_full_state, None, class_with_globalize_methods)
        setattr(class_with_globalize_methods, name, new_method)
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
            raise TypeError("property decorator has to be given first.")

        func._map_to_full_state_def_arg = self.default_arg
        func._map_to_full_state_along_axis = self.extend_along_axis
        return func
