# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
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

r"""
Python interface to utility functions
"""

import numpy as _np
cimport numpy as _np

__all__ = []

cdef extern from "_util.h":
    int _get_therm_state_break_points(int *T_x, int seq_length, int *break_points)

def get_therm_state_break_points(
    _np.ndarray[int, ndim=1, mode="c"] T_x not None):
    r"""
    Find thermodynamic state changes within a trajectory

    Parameters
    ----------
    T_x : numpy.ndarray(shape=(X), dtype=numpy.intc)
        Thermodynamic state sequence of a trajectory of length X

    Returns
    -------
    T_B : numpy.ndarray(shape=(B), dtype=numpy.intc)
        Sequence of first subsequence starting frames
    """
    T_B = _np.zeros(shape=(T_x.shape[0],), dtype=_np.intc)
    nb = _get_therm_state_break_points(
        <int*> _np.PyArray_DATA(T_x),
        T_x.shape[0],
        <int*> _np.PyArray_DATA(T_B))
    return _np.ascontiguousarray(T_B[:nb])


