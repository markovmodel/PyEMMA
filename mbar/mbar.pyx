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
Python interface to the TRAM estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np

__all__ = ['iterate_fk']

cdef extern from "_mbar.h":
    void _iterate_fk(
        double *log_N_K, double *f_K, double *b_K_x,
        int n_therm_states, int seq_length, double *scratch_T, double *new_f_K)
    void _get_fi(
        double *log_N_K, double *f_K, double *b_K_x, int * M_x,
        int n_therm_states, int n_markov_states, int seq_length, double *scratch_T, double *f_i)
    void _normalize_fi(
        double *f_K, double *f_i, int n_therm_states, int n_markov_states, double *scratch_M)

def iterate_fk(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_x not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] new_f_K not None):
    r"""
    Calculate the reduced thermodynamic free energies f_K
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    new_f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    _iterate_fk(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(b_K_x),
        b_K_x.shape[0],
        b_K_x.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_f_K))

def get_fi(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_x not None,
    _np.ndarray[int, ndim=1, mode="c"] M_x not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    n_discrete_states):
    r"""
    Calculate the reduced free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    new_f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    f_i = _np.zeros(shape=(n_discrete_states,), dtype=_np.float64)
    _get_fi(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(b_K_x),
        <int*> _np.PyArray_DATA(M_x),
        b_K_x.shape[0],
        n_discrete_states,
        b_K_x.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(f_i))
    return f_i

def normalize_fi(
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Calculate the reduced free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    new_f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    _normalize_fi(
        <double*> _np.PyArray_DATA(f_i),
        <double*> _np.PyArray_DATA(f_K),
        f_K.shape[0],
        f_i.shape[0],
        <double*> _np.PyArray_DATA(scratch_M))
