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

import numpy as np
cimport numpy as np

cdef extern from "_lse.h":
    double _logsumexp(double *array, int length)
    double _logsumexp_pair(double a, double b)

def logsumexp(np.ndarray[double, ndim=1, mode="c"] array not None):
    return _logsumexp(<double*> np.PyArray_DATA(array), array.shape[0])

def logsumexp_pair(a, b):
    return _logsumexp_pair(a, b)
