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
Python interface to the dTRAM estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np

__all__ = [
    'set_lognu',
    'iterate_lognu',
    'iterate_fi',
    'get_pk',
    'get_p',
    'get_fk']

cdef extern from "_dtram.h":
    void _set_lognu(
        double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states)
    void _iterate_lognu(
        double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
        int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
    void _iterate_fi(
        double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
        int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i)
    void _get_p(
        double *log_nu_i, double *b_i, double *f_i, int *C_ij,
        int n_markov_states, double *scratch_M, double *p_ij)
    void _get_fk(
        double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
        double *scratch_M, double *f_K)

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
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_nu_K_i not None):
    r"""
    Update the logarithms of the Lagrangian multipliers

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the log of the Lagrangian multipliers
    """
    _iterate_lognu(
        <double*> _np.PyArray_DATA(log_nu_K_i),
        <double*> _np.PyArray_DATA(b_K_i),
        <double*> _np.PyArray_DATA(f_i),
        <int*> _np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(new_log_nu_K_i))

def iterate_fi(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] new_f_i not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_TM : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        scratch array for logsumexp operations
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        target array for the reduced unbiased free energies
    """
    _iterate_fi(
        <double*> _np.PyArray_DATA(log_nu_K_i),
        <double*> _np.PyArray_DATA(b_K_i),
        <double*> _np.PyArray_DATA(f_i),
        <int*> _np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_TM),
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(new_f_i))

def get_pk(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    p_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.float64)
        transition matrices for all thermodynamic states
    """
    p_K_ij = _np.zeros(shape=(C_K_ij.shape[0], C_K_ij.shape[1], C_K_ij.shape[2]), dtype=_np.float64)
    for K in range(log_nu_K_i.shape[0]):
        p_K_ij[K, :, :] = get_p(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_M, K)[:, :]
    return p_K_ij

def get_p(
    _np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    therm_state):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    therm_state : int
        target thermodynamic state

    Returns
    -------
    p_ij : numpy.ndarray(shape=(M, M), dtype=numpy.float64)
        transition matrix for the target thermodynamic state
    """
    p_ij = _np.zeros(shape=(f_i.shape[0], f_i.shape[0]), dtype=_np.float64)
    _get_p(
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(log_nu_K_i[therm_state, :])),
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(b_K_i[therm_state, :])),
        <double*> _np.PyArray_DATA(f_i),
        <int*> _np.PyArray_DATA(_np.ascontiguousarray(C_K_ij[therm_state, :, :])),
        f_i.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(p_ij))
    return p_ij

def get_fk(
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    f_K : numpy.ndarray(shape=(T,), dtype=numpy.float64)
        target array for the reduced thermodynamic free energies
    """
    _get_fk(
        <double*> _np.PyArray_DATA(b_K_i),
        <double*> _np.PyArray_DATA(f_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(f_K))
