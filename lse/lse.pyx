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

import numpy as _np
cimport numpy as _np

__all__ = ['logsumexp', 'logsumexp_pair']

cdef extern from "_lse.h":
    double _logsumexp(double *array, int length)
    double _logsumexp_pair(double a, double b)

def logsumexp(_np.ndarray[double, ndim=1, mode="c"] array not None):
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

    Notes
    -----
    The logsumexp() function returns

    .. math:
        \ln\left( \sum_{i=0}^{n-1} \exp(a_i) \right)

    where the :math:`a_i` are the :math:`n` values in the supplied array.
    """
    return _logsumexp(<double*> _np.PyArray_DATA(array.copy()), array.shape[0])

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

    Notes
    -----
    The logsumexp_pair() function returns

    .. math:
        \ln\left( \exp(a) + \exp(b) \right)

    where the :math:`a` and :math:`b` are the supplied values.
    """
    return _logsumexp_pair(a, b)
