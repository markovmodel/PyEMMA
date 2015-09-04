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

import numpy as np
cimport numpy as np

__all__ = [
    'dtram_set_lognu',
    'dtram_lognu',
    'dtram_fi',
    'dtram_pk',
    'dtram_p',
    'dtram_fk']

cdef extern from "_dtram.h":
    void _dtram_set_lognu(
        double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states)
    void _dtram_lognu(
        double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
        int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
    void _dtram_fi(
        double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
        int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i)
    void _dtram_pk(
        double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
        int n_markov_states, double *scratch_M, double *p_K_ij)
    void _dtram_p(
        double *log_nu_i, double *b_i, double *f_i, int *C_ij,
        int n_markov_states, double *scratch_M, double *p_ij)
    void _dtram_fk(
        double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
        double *scratch_M, double *f_K)

def dtram_set_lognu(
    np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    np.ndarray[int, ndim=3, mode="c"] C_K_ij not None):
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
    _dtram_set_lognu(
        <double*> np.PyArray_DATA(log_nu_K_i),
        <int*> np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1])

def dtram_lognu(
    np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=2, mode="c"] new_log_nu_K_i not None):
    r"""
    Update the logarithms of the Lagrangian multipliers

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=np.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=np.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=np.float64)
        scratch array for logsumexp operations
    new_log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=np.float64)
        target array for the log of the Lagrangian multipliers
    """
    _dtram_lognu(
        <double*> np.PyArray_DATA(log_nu_K_i),
        <double*> np.PyArray_DATA(b_K_i),
        <double*> np.PyArray_DATA(f_i),
        <int*> np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(new_log_nu_K_i))

def dtram_fi(
    np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=1, mode="c"] new_f_i not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=np.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=np.intc)
        multistate count matrix
    scratch_TM : numpy.ndarray(shape=(T, M), dtype=np.float64)
        scratch array for logsumexp operations
    scratch_M : numpy.ndarray(shape=(M,), dtype=np.float64)
        scratch array for logsumexp operations
    new_f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        target array for the reduced unbiased free energies
    """
    _dtram_fi(
        <double*> np.PyArray_DATA(log_nu_K_i),
        <double*> np.PyArray_DATA(b_K_i),
        <double*> np.PyArray_DATA(f_i),
        <int*> np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_TM),
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(new_f_i))

def dtram_pk(
    np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=3, mode="c"] p_K_ij not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_nu_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_K_i : numpy.ndarray(shape=(T, M), dtype=np.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        reduced unbiased free energies
    C_K_ij : numpy.ndarray(shape=(T, M, M), dtype=np.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=np.float64)
        scratch array for logsumexp operations
    p_K_ij : numpy.ndarray(shape=(T, M, M), dtype=np.float64)
        target array for the transition matrices
    """
    _dtram_pk(
        <double*> np.PyArray_DATA(log_nu_K_i),
        <double*> np.PyArray_DATA(b_K_i),
        <double*> np.PyArray_DATA(f_i),
        <int*> np.PyArray_DATA(C_K_ij),
        log_nu_K_i.shape[0],
        log_nu_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(p_K_ij))

def dtram_p(
    np.ndarray[double, ndim=1, mode="c"] log_nu_i not None,
    np.ndarray[double, ndim=1, mode="c"] b_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[int, ndim=2, mode="c"] C_ij not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=2, mode="c"] p_ij not None):
    r"""
    Compute the transition matrices for a single thermodynamic state

    Parameters
    ----------
    log_nu_i : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        log of the Lagrangian multipliers
    b_i : numpy.ndarray(shape=(M,), dtype=np.intc)
        reduced bias energies of the M markov states in the desired thermodynamic state
    f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        reduced unbiased free energies
    C_ij : numpy.ndarray(shape=(M, M), dtype=np.intc)
        count matrix in the desired thermodynamic state
    scratch_M : numpy.ndarray(shape=(M,), dtype=np.float64)
        scratch array for logsumexp operations
    p_ij : numpy.ndarray(shape=(M, M), dtype=np.float64)
        target array for the transition matrix in the desired thermodynamic state
    """
    _dtram_p(
        <double*> np.PyArray_DATA(log_nu_i),
        <double*> np.PyArray_DATA(b_i),
        <double*> np.PyArray_DATA(f_i),
        <int*> np.PyArray_DATA(C_ij),
        log_nu_i.shape[0],
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(p_ij))

def dtram_fk(
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    b_K_i : numpy.ndarray(shape=(T, M), dtype=np.intc)
        reduced bias energies of the T thermodynamic and M markov states
    f_i : numpy.ndarray(shape=(M,), dtype=np.float64)
        reduced unbiased free energies
    scratch_M : numpy.ndarray(shape=(M,), dtype=np.float64)
        scratch array for logsumexp operations
    f_K : numpy.ndarray(shape=(T,), dtype=np.float64)
        target array for the reduced thermodynamic free energies
    """
    _dtram_fk(
        <double*> np.PyArray_DATA(b_K_i),
        <double*> np.PyArray_DATA(f_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(f_K))
