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
Python interface to the WHAM estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np

__all__ = ['iterate_fi', 'iterate_fk']

cdef extern from "_wham.h":
    void _iterate_fi(
        double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
        int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T, double *f_i)
    void _iterate_fk(
        double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
        double *scratch_M, double *f_K)

def iterate_fi(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=1, mode="c"] log_N_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None):
    r"""
    Calculate the reduced free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    log_N_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        log of the state counts in each of the M markov states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The iterate_fi() function computes

    .. math:
        f_i = -\ln\left( \frac{
                \sum_{K=0}^{N_T-1} N_i^{(K)}
            }{
                \sum_{K=0}^{N_T-1} \exp(f^{(K)}-b_i^{(K)}) \sum_{j=0}^{N_M-1} N_j^{(K)}
            }\right)

    which equals to

    .. math:
        f_i = \ln\left(
                \sum_{K=0}^{N_T-1} \exp\left(
                    f^{(K)} - b_i^{(K)} + \ln\left( \sum_{j=0}^{N_M-1} N_j^{(K)} \right)
                \right)
            \right) - \ln\left( \sum_{K=0}^{N_T-1} N_i^{(K)} \right)

    in the logsumexp scheme. Afterwards, we apply

    .. math:
        f_i \leftarrow  f_i + \ln\left(
                \sum_{j=0}^{N_M-1} \exp(-f_j)
            \right)

    which is equilvalent to a renormalisation of the stationary distribution.
    """
    _iterate_fi(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(log_N_i),
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(f_i))

def iterate_fk(
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Calculate the reduced thermodynamic free energies f_K
        
    Parameters
    ----------
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M markov states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The iterate_fk() function computes

    .. math:
        f^{(K)} = -\ln\left(
                \sum_{i=0}^{N_M-1} \exp(-b_i^{(K)}-f_i)
            \right)

    which is already in the logsumexp form.
    """
    _iterate_fk(
        <double*> _np.PyArray_DATA(f_i),
        <double*> _np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(f_K))
