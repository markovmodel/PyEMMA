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
        double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,
        int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
    void _update_biased_conf_energies(
        double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, double *bias_energy_sequence,
        int *state_sequence, int *state_counts, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
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
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
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
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    p_K_ij : numpy.ndarray(shape=(T, M, M), dtype=numpy.float64)
        transition matrices for all thermodynamic states
    """
    p_K_ij = _np.zeros(shape=(count_matrices.shape[0], count_matrices.shape[1], count_matrices.shape[2]), dtype=_np.float64)
    for K in range(log_lagrangian_mult.shape[0]):
        p_K_ij[K, :, :] = estimate_transition_matrix(log_lagrangian_mult, biased_conf_energies, count_matrices, scratch_M, K)[:, :]
    return p_K_ij

def estimate_transition_matrix(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
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
    scratch_M : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        scratch array for logsumexp operations
    therm_state : int
        target thermodynamic state

    Returns
    -------
    transition_matrix : numpy.ndarray(shape=(M, M), dtype=numpy.float64)
        transition matrix for the target thermodynamic state
    """
    transition_matrix = _np.zeros(shape=(biased_conf_energies.shape[1], biased_conf_energies.shape[1]), dtype=_np.float64)
    _estimate_transition_matrix(
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(log_lagrangian_mult[therm_state, :])),
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(biased_conf_energies[therm_state, :])),
        <int*> _np.PyArray_DATA(_np.ascontiguousarray(count_matrices[therm_state, :, :])),
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(transition_matrix))
    return transition_matrix

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
    old_biased_conf_energies = biased_conf_energies.copy()
    old_log_lagrangian_mult = log_lagrangian_mult.copy()
    for _m in range(maxiter):
        update_lagrangian_mult(old_log_lagrangian_mult, biased_conf_energies, count_matrices, scratch_M, log_lagrangian_mult)
        update_biased_conf_energies(log_lagrangian_mult, old_biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
            state_counts, log_R_K_i, scratch_M, scratch_T, biased_conf_energies)
        if _np.max(_np.abs(biased_conf_energies - old_biased_conf_energies)) < maxerr:
            break
        else:
            old_biased_conf_energies[:] = biased_conf_energies[:]
            old_log_lagrangian_mult[:] = log_lagrangian_mult[:]
    conf_energies = get_conf_energies(bias_energy_sequence, state_sequence, log_R_K_i, scratch_M, scratch_T)
    therm_energies = get_therm_energies(biased_conf_energies, scratch_M)
    normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)
    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult
