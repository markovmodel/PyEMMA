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
Python interface to the MBAR estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np

__all__ = ['update_therm_energies', 'normalize', 'get_fi', 'estimate']

cdef extern from "_mbar.h":
    void _update_therm_energies(
        double *log_N_K, double *f_K, double *b_K_x,
        int n_therm_states, int seq_length, double *scratch_T, double *new_f_K)
    void _normalize(
        double *log_N_K, double *b_K_x, int n_therm_states, int seq_length,
        double *scratch_T, double *f_K)
    void _get_fi(
        double *log_N_K, double *f_K, double *b_K_x, int * M_x,
        int n_therm_states, int n_markov_states, int seq_length,
        double *scratch_M, double *scratch_T, double *f_i)

def update_therm_energies(
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
    _update_therm_energies(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(b_K_x),
        b_K_x.shape[0],
        b_K_x.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_f_K))

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_x not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Shift the reduced thermodynamic free energies f_K such that the unbiased thermodynamic
    free energy is zero
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
        scratch array
    """
    _normalize(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(b_K_x),
        b_K_x.shape[0],
        b_K_x.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(f_K))

def get_fi(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_x not None,
    _np.ndarray[int, ndim=1, mode="c"] M_x not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    n_discrete_states):
    r"""
    Calculate the reduced unbiased free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    M_x : numpy.ndarray(shape=(X), dtype=numpy.intc)
        discrete states indices for all X samples
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    n_discrete_states : int
        number of discrete states (M)

    Returns
    -------
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies
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
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(f_i))
    return f_i

def estimate(N_K, b_K_x, maxiter=1000, maxerr=1.0E-8, f_K=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    N_K : numpy.ndarray(shape=(T), dtype=numpy.intc)
        discrete state counts in the T thermodynamic states
    b_K_x : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states

    Returns
    -------
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    """
    T = N_K.shape[0]
    log_N_K = _np.log(N_K)
    if f_K is None:
        f_K = _np.zeros(shape=(T,), dtype=_np.float64)
    old_f_K = f_K.copy()
    scratch = _np.zeros(shape=(T,), dtype=_np.float64)
    stop = False
    for _m in range(maxiter):
        update_therm_energies(log_N_K, old_f_K, b_K_x, scratch, f_K)
        if _np.max(_np.abs((f_K - old_f_K))) < maxerr:
            stop = True
        else:
            old_f_K[:] = f_K[:]
        if stop:
            break
    normalize(log_N_K, b_K_x, scratch, f_K)
    return f_K
