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

__all__ = [
    'set_lognu',
    'iterate_lognu',
    'iterate_fki']

cdef extern from "_tram.h":
    void _set_lognu(double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states)
    void _iterate_lognu(
        double *log_nu_K_i, double *f_K_i, int *C_K_ij,
        int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
    void _iterate_fki(
        double *log_nu_K_i, double *f_K_i, int *C_K_ij, double *b_K_x,
        int *M_x, int *N_K_i, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T,
        double *new_f_K_i, int K_target)

def set_lognu(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None):
    r"""
    Set the logarithm of the Lagrangian multipliers with an initial guess based
    on the transition counts

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers (allocated but unset)
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    """
    _set_lognu(
        <double*> _np.PyArray_DATA(log_nu_K_i),
        <int*> _np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1])

def iterate_lognu(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] f_K_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_nu_K_i not None):
    r"""
    Update the logarithms of the Lagrangian multipliers

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    f_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the log of the Lagrangian multipliers
    """
    _iterate_lognu(
        <double*> _np.PyArray_DATA(log_nu_K_i),
        <double*> _np.PyArray_DATA(f_K_i),
        <int*> _np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(new_log_nu_K_i))

def iterate_fki(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] f_K_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_x not None,
    _np.ndarray[int, ndim=1, mode="c"] M_x not None,
    _np.ndarray[int, ndim=2, mode="c"] N_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] log_R_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=2, mode="c"] new_f_K_i not None,
    target_therm_state=0):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    f_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_f_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the reduced free energies
    target_therm_state : int
        references the thermodynamic target state
    """
    _iterate_fki(
        <double*> _np.PyArray_DATA(log_nu_K_i),
        <double*> _np.PyArray_DATA(f_K_i),
        <int*> _np.PyArray_DATA(C_K_ij),
        <double*> _np.PyArray_DATA(b_K_x),
        <int*> _np.PyArray_DATA(M_x),
        <int*> _np.PyArray_DATA(N_K_i),
        M_x.shape[0],
        <double*> _np.PyArray_DATA(log_R_K_i),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_f_K_i),
        target_therm_state)
