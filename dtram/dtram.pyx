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
    'init_lagrangian_mult',
    'update_lagrangian_mult',
    'update_conf_energies',
    'estimate_transition_matrices',
    'estimate_transition_matrix',
    'get_therm_energies',
    'normalize',
    'estimate']

cdef extern from "_dtram.h":
    void _init_lagrangian_mult(
        int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
    void _update_lagrangian_mult(
        double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices, int n_therm_states,
        int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
    void _update_conf_energies(
        double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices, int n_therm_states,
        int n_conf_states, double *scratch_TM, double *new_conf_energies)
    void _estimate_transition_matrix(
        double *log_lagrangian_mult, double *b_i, double *conf_energies, int *count_matrix,
        int n_conf_states, double *scratch_M, double *transition_matrix)
    void _get_therm_energies(
        double *bias_energies, double *conf_energies, int n_therm_states, int n_conf_states,
        double *scratch_M, double *therm_energies)
    void _normalize(
        int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies, double *conf_energies)

def init_lagrangian_mult(
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None):
    r"""
    Set the logarithm of the Lagrangian multipliers with an initial guess based
    on the transition counts

    Parameters
    ----------
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers (allocated but unset)
    """
    _init_lagrangian_mult(
        <int*> _np.PyArray_DATA(count_matrices),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(log_lagrangian_mult))

def update_lagrangian_mult(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_lagrangian_mult not None):
    r"""
    Update the logarithms of the Lagrangian multipliers

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the log of the Lagrangian multipliers
    """
    _update_lagrangian_mult(
        <double*> _np.PyArray_DATA(log_lagrangian_mult),
        <double*> _np.PyArray_DATA(bias_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(new_log_lagrangian_mult))

def update_conf_energies(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=1, mode="c"] new_conf_energies not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_TM : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        target array for the reduced unbiased free energies
    """
    _update_conf_energies(
        <double*> _np.PyArray_DATA(log_lagrangian_mult),
        <double*> _np.PyArray_DATA(bias_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_TM),
        <double*> _np.PyArray_DATA(new_conf_energies))

def estimate_transition_matrices(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    transition_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.float64)
        transition matrices for all thermodynamic states
    """
    transition_matrices = _np.zeros(shape=(count_matrices.shape[0], count_matrices.shape[1], count_matrices.shape[2]), dtype=_np.float64)
    for K in range(log_lagrangian_mult.shape[0]):
        transition_matrices[K, :, :] = estimate_transition_matrix(log_lagrangian_mult, bias_energies, conf_energies, count_matrices, scratch_M, K)[:, :]
    return transition_matrices

def estimate_transition_matrix(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    therm_state):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    therm_state : int
        target thermodynamic state

    Returns
    -------
    transition_matrix : numpy.ndarray(shape=(M, M), dtype=numpy.float64)
        transition matrix for the target thermodynamic state
    """
    transition_matrix = _np.zeros(shape=(conf_energies.shape[0], conf_energies.shape[0]), dtype=_np.float64)
    _estimate_transition_matrix(
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(log_lagrangian_mult[therm_state, :])),
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(bias_energies[therm_state, :])),
        <double*> _np.PyArray_DATA(conf_energies),
        <int*> _np.PyArray_DATA(_np.ascontiguousarray(count_matrices[therm_state, :, :])),
        conf_energies.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(transition_matrix))
    return transition_matrix

def get_therm_energies(
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M markov states
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased free energies
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T,), dtype=numpy.float64)
        reduced thermodynamic free energies
    """
    therm_energies = _np.zeros(shape=(bias_energies.shape[0],), dtype=_np.float64)
    _get_therm_energies(
        <double*> _np.PyArray_DATA(bias_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        bias_energies.shape[0],
        bias_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies))
    return therm_energies

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None):
    r"""
    Normalize the unbiased reduced free energies and shift the reduced thermodynamic
    free energies accordingly

    Parameters
    ----------
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.intc)
        reduced thermodynamic free energies
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies
    """
    _normalize(
        therm_energies.shape[0],
        conf_energies.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies))

def estimate(count_matrices, bias_energies, maxiter=1000, maxerr=1.0E-8, log_lagrangian_mult=None, conf_energies=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        transition count matrices for all T thermodynamic states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic and M discrete states
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced unbiased free energies of the M discrete states
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64), OPTIONAL
        initial guess for the logarithm of the Lagrangian multipliers

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies of the M discrete states
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        logarithm of the Lagrangian multipliers
    """
    if log_lagrangian_mult is None:
        log_lagrangian_mult = _np.zeros(shape=bias_energies.shape, dtype=_np.float64)
        init_lagrangian_mult(count_matrices, log_lagrangian_mult)
    if conf_energies is None:
        conf_energies = _np.zeros(shape=bias_energies.shape[1], dtype=_np.float64)
    scratch_TM = _np.zeros(shape=bias_energies.shape, dtype=_np.float64)
    scratch_M = _np.zeros(shape=conf_energies.shape, dtype=_np.float64)
    old_log_lagrangian_mult = log_lagrangian_mult.copy()
    old_conf_energies = conf_energies.copy()
    for m in range(maxiter):
        update_lagrangian_mult(old_log_lagrangian_mult, bias_energies, conf_energies, count_matrices, scratch_M, log_lagrangian_mult)
        update_conf_energies(log_lagrangian_mult, bias_energies, old_conf_energies, count_matrices, scratch_TM, conf_energies)
        delta_log_lagrangian_mult = _np.max(_np.abs((log_lagrangian_mult - old_log_lagrangian_mult)))
        delta_conf_energies = _np.max(_np.abs((conf_energies - old_conf_energies)))
        if delta_log_lagrangian_mult < maxerr and delta_conf_energies < maxerr:
            stop = True
        else:
            old_log_lagrangian_mult[:] = log_lagrangian_mult[:]
            old_conf_energies[:] = conf_energies[:]
    therm_energies = get_therm_energies(bias_energies, conf_energies, scratch_M)
    normalize(scratch_M, therm_energies, conf_energies)
    return therm_energies, conf_energies, log_lagrangian_mult
