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
import sys

__all__ = [
    'init_lagrangian_mult',
    'update_lagrangian_mult',
    'update_biased_conf_energies',
    'get_conf_energies',
    'normalize',
    'estimate_transition_matrix',
    'estimate_transition_matrices',
    'estimate']

cdef extern from "_tram.h":
    void _init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
    void _update_lagrangian_mult(
        double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,  int* state_counts,
        int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
    void _update_biased_conf_energies(
        double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, double *bias_energy_sequence,
        int *state_sequence, int *state_counts, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_conf_states, int do_shift, double *scratch_M, double *scratch_T,
        double *new_biased_conf_energies)
    void _get_conf_energies(
        double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
        double *conf_energies)
    void _get_therm_energies(
        double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies)
    void _normalize(
        double *conf_energies, double *biased_conf_energies, double *therm_energies,
        int n_therm_states, int n_conf_states, double *scratch_M)
    void _estimate_transition_matrix(
        double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
        int n_conf_states, double *scratch_M, double *transition_matrix)
    double _log_likelihood_assuming_fulfilled_constraints(
        double *old_log_lagrangian_mult, double *new_log_lagrangian_mult,
        double *old_biased_conf_energies, double *new_biased_conf_energies,
        int *count_matrices, int *state_counts,
        int n_therm_states, int n_conf_states,
        double *bias_energy_sequence, int *state_sequence, int seq_length,
        double *scratch_T, double *scratch_M, double *scratch_TM, double *scratch_MM)


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
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_lagrangian_mult not None):
    r"""
    Update the logarithms of the Lagrangian multipliers

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    new_log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the log of the Lagrangian multipliers
    """
    _update_lagrangian_mult(
        <double*> _np.PyArray_DATA(log_lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(new_log_lagrangian_mult))

def update_biased_conf_energies(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] log_R_K_i not None,
    do_shift,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_energies not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequence : numpy.ndarray(shape=(X,), dtype=numpy.intc)
        Markov state indices for all X samples
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        number of visits to thermodynamic state K and Markov state i
    log_R_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        scratch array for sum of TRAM log pseudo-counts and biased_conf_energies
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations
    do_shift : shift new_biased_conf_energies s.t. the minimum is zero
    new_biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for the reduced free energies
    """
    _update_biased_conf_energies(
        <double*> _np.PyArray_DATA(log_lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        <int*> _np.PyArray_DATA(state_sequence),
        <int*> _np.PyArray_DATA(state_counts),
        state_sequence.shape[0],
        <double*> _np.PyArray_DATA(log_R_K_i),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        int(do_shift),
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_biased_conf_energies))

def get_conf_energies(
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[double, ndim=2, mode="c"] log_R_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequence : numpy.ndarray(shape=(X,), dtype=numpy.intc)
        Markov state indices for all X samples
    log_R_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        precomputed sum of TRAM log pseudo-counts and biased_conf_energies
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        unbiased (Markov) free energies
    """
    # later this can be extended to other thermodynmic states and
    # arbitrary expectations (not only pi)
    conf_energies = _np.zeros(shape=(log_R_K_i.shape[1],), dtype=_np.float64)
    _get_conf_energies(
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        <int*> _np.PyArray_DATA(state_sequence),
        state_sequence.shape[0],
        <double*> _np.PyArray_DATA(log_R_K_i),
        log_R_K_i.shape[0],
        log_R_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(conf_energies))
    return conf_energies

def get_therm_energies(
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    biased_conf_energies : numpy.ndarray(shape=(T, X), dtype=numpy.intc)
        reduced discrete state free energies for all T thermodynamic states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced thermodynamic free energies
    """
    therm_energies = _np.zeros(shape=(biased_conf_energies.shape[0],), dtype=_np.float64)
    _get_therm_energies(
        <double*> _np.PyArray_DATA(biased_conf_energies),
        biased_conf_energies.shape[0],
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies))
    return therm_energies

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Update the reduced unbiased free energies

    Parameters
    ----------
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.intc)
        unbiased reduced bias energies in the M discrete states
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic and M discrete states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.intc)
        reduced thermodynamic free energies
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    """
    _normalize(
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(biased_conf_energies),
        <double*> _np.PyArray_DATA(therm_energies),
        biased_conf_energies.shape[0],
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M))

def estimate_transition_matrices(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix

    Returns
    -------
    p_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.float64)
        transition matrices for all thermodynamic states
    """
    if scratch_M is None:
        scratch_M = _np.zeros(shape=(count_matrices.shape[1],), dtype=_np.float64)
    p_K_ij = _np.zeros(shape=(count_matrices.shape[0], count_matrices.shape[1], count_matrices.shape[2]), dtype=_np.float64)
    for K in range(log_lagrangian_mult.shape[0]):
        p_K_ij[K, :, :] = estimate_transition_matrix(log_lagrangian_mult, biased_conf_energies, count_matrices, scratch_M, K)[:, :]
    return p_K_ij

def estimate_transition_matrix(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M,
    therm_state):
    r"""
    Compute the transition matrices for all thermodynamic states

    Parameters
    ----------
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced unbiased free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    therm_state : int
        target thermodynamic state

    Returns
    -------
    transition_matrix : numpy.ndarray(shape=(M, M), dtype=numpy.float64)
        transition matrix for the target thermodynamic state
    """
    if scratch_M is None:
        scratch_M = _np.zeros(shape=(count_matrices.shape[1],), dtype=_np.float64)
    transition_matrix = _np.zeros(shape=(biased_conf_energies.shape[1], biased_conf_energies.shape[1]), dtype=_np.float64)
    _estimate_transition_matrix(
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(log_lagrangian_mult[therm_state, :])),
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(biased_conf_energies[therm_state, :])),
        <int*> _np.PyArray_DATA(_np.ascontiguousarray(count_matrices[therm_state, :, :])),
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(transition_matrix))
    return transition_matrix

def log_likelihood_assuming_fulfilled_constraints(
    _np.ndarray[double, ndim=2, mode="c"] old_log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] old_biased_conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_MM not None):
    r"""
    Computes the TRAM log-likelihood with an arbitrary mu that is implicitly
    given by old_log_lagrangian_mult and old_biased_conf_energies.

    new_biased_conf_energies is assumed to be the correct normalization
    of mu. new_log_lagrangian_mult is assumed to code for a T matrix
    that fulfills detailled balance w.r.t. new_biased_conf_energies.

    When the TRAM iteration is converged, inserting the same values for
    old* and new* will yield the actual log-likelihood. Otherwise this
    just serves as a helper function for log_likelihood.
    """

    logL = _log_likelihood_assuming_fulfilled_constraints(
        <double*> _np.PyArray_DATA(old_log_lagrangian_mult),
        <double*> _np.PyArray_DATA(new_log_lagrangian_mult),
        <double*> _np.PyArray_DATA(old_biased_conf_energies),
        <double*> _np.PyArray_DATA(new_biased_conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        state_counts.shape[0],
        state_counts.shape[1],
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        <int*> _np.PyArray_DATA(state_sequence),
        state_sequence.shape[0],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_TM),
        <double*> _np.PyArray_DATA(scratch_MM))

    return logL

def log_likelihood(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_MM not None,
    maxerr):
    r"""Computes the TRAM log-likelihood with an arbitrary mu that is implicitly
       given by log_lagrangian_mult and biased_conf_energies.
    """

    # Compute the normalization contant of mu that is implicitly given
    # by (old_)log_lagrangian_mult and (old_)biased_conf_energies.
    very_old_log_lagrangian_mult = log_lagrangian_mult.copy()
    old_log_lagrangian_mult = log_lagrangian_mult.copy()
    old_biased_conf_energies = biased_conf_energies.copy()
    new_biased_conf_energies = _np.zeros_like(biased_conf_energies)
    update_biased_conf_energies(old_log_lagrangian_mult, old_biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
                                state_counts, scratch_TM, False, scratch_M, scratch_T, new_biased_conf_energies)
    # Compute a normalized T matrix that fulfills detailled balance w.r.t.
    # new_biased_conf_energies by iterating the Lagrange multipliers.
    new_log_lagrangian_mult = _np.zeros_like(old_log_lagrangian_mult)
    while True:
        update_lagrangian_mult(old_log_lagrangian_mult, new_biased_conf_energies, count_matrices, state_counts, scratch_M, new_log_lagrangian_mult)
        nz = _np.where(_np.logical_and(_np.logical_not(_np.isinf(new_log_lagrangian_mult)),
                                       _np.logical_not(_np.isinf(old_log_lagrangian_mult))))
        if _np.max(_np.abs(new_log_lagrangian_mult[nz] - old_log_lagrangian_mult[nz])) < maxerr:
            break
        old_log_lagrangian_mult[:] = new_log_lagrangian_mult[:]
    # Use helper function.
    logL = log_likelihood_assuming_fulfilled_constraints(
               very_old_log_lagrangian_mult, new_log_lagrangian_mult,
               old_biased_conf_energies, new_biased_conf_energies,
               count_matrices, bias_energy_sequence, state_sequence, state_counts,
               scratch_M, scratch_T, scratch_TM, scratch_MM)
    return logL

def estimate(count_matrices, state_counts, bias_energy_sequence, state_sequence, maxiter=1000, maxerr=1.0E-8, biased_conf_energies=None, log_lagrangian_mult=None):
    r"""
    Estimate the reduced discrete state free energies and thermodynamic free energies
        
    Parameters
    ----------
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        transition count matrices for all T thermodynamic states
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        state counts for all M discrete and T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequence : numpy.ndarray(shape=(X), dtype=numpy.float64)
        discrete state indices for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced discrete state free energies for all T thermodynamic states
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64), OPTIONAL
        initial guess for the logarithm of the Lagrangian multipliers

    Returns
    -------
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced discrete state free energies for all T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased discrete state free energies
    therm_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced thermodynamic free energies
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        logarithm of the Lagrangian multipliers
    """
    if biased_conf_energies is None:
        biased_conf_energies = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    if log_lagrangian_mult is None:
        log_lagrangian_mult = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
        init_lagrangian_mult(count_matrices, log_lagrangian_mult)
    log_R_K_i = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    scratch_T = _np.zeros(shape=(count_matrices.shape[0],), dtype=_np.float64)
    scratch_M = _np.zeros(shape=(count_matrices.shape[1],), dtype=_np.float64)
    scratch_TM = _np.zeros(shape=count_matrices.shape[0:2], dtype=_np.float64)
    scratch_MM = _np.zeros(shape=count_matrices.shape[1:3], dtype=_np.float64)
    old_biased_conf_energies = biased_conf_energies.copy()
    old_log_lagrangian_mult = log_lagrangian_mult.copy()

    if True:
        logL_0 = log_likelihood(log_lagrangian_mult, biased_conf_energies,
                     count_matrices, bias_energy_sequence, state_sequence, state_counts,
                     scratch_M, scratch_T, scratch_TM, scratch_MM, maxerr)
        logL_hist = [logL_0]
    else:
        logL_hist = []
    
    for _m in range(maxiter):
        update_lagrangian_mult(old_log_lagrangian_mult, biased_conf_energies, count_matrices, state_counts, scratch_M, log_lagrangian_mult)
        update_biased_conf_energies(log_lagrangian_mult, old_biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
            state_counts, log_R_K_i, True, scratch_M, scratch_T, biased_conf_energies)
        if _m%10 == 0:
            logL = log_likelihood_assuming_fulfilled_constraints( # better just use log_likelihood?
                       log_lagrangian_mult, log_lagrangian_mult,
                       biased_conf_energies, biased_conf_energies,
                       count_matrices, bias_energy_sequence, state_sequence,
                       state_counts, scratch_M, scratch_T, scratch_TM, scratch_MM)
            logL_hist.append(logL)
            print>>sys.stderr, logL, 
            print>>sys.stderr, _np.max(_np.abs(biased_conf_energies - old_biased_conf_energies))
        if _np.max(_np.abs(biased_conf_energies - old_biased_conf_energies)) < maxerr:
            break
        else:
            old_biased_conf_energies[:] = biased_conf_energies[:]
            old_log_lagrangian_mult[:] = log_lagrangian_mult[:]
    conf_energies = get_conf_energies(bias_energy_sequence, state_sequence, log_R_K_i, scratch_M, scratch_T)
    therm_energies = get_therm_energies(biased_conf_energies, scratch_M)
    normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)
    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult #, logL_hist
