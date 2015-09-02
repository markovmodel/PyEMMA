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
Python interface to the logsumexp summation scheme for numerically robust summation of exponentials.
"""

import numpy as np
cimport numpy as np

__all__ = ['logsumexp', 'logsumexp_pair']

cdef extern from "_lse.h":
    double _logsumexp(double *array, int length)
    double _logsumexp_pair(double a, double b)

def logsumexp(np.ndarray[double, ndim=1, mode="c"] array not None):
    r"""
    Perform a summation of an array of exponentials via the logsumexp scheme
        
    Parameters
    ----------
    array : numpy.ndarray(dtype=numpy.float64)
        arguments of the exponentials

    Returns
    -------
    ln_sum : float
        logarithm of the sum of exponentials
    """
    return _logsumexp(<double*> np.PyArray_DATA(array), array.shape[0])

def logsumexp_pair(a, b):
    r"""
    Perform a summation of two exponentials via the logsumexp scheme
        
    Parameters
    ----------
    a : float
        arguments of the first exponential
    b : float
        arguments of the second exponential

    Returns
    -------
    ln_sum : float
        logarithm of the sum of exponentials
    """
    return _logsumexp_pair(a, b)
