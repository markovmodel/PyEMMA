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
from .callback import CallbackInterrupt

__all__ = [
    'init_lagrangian_mult',
    'update_lagrangian_mult',
    'update_biased_conf_energies',
    'get_conf_energies',
    'normalize',
    'get_pointwise_unbiased_free_energies',
    'estimate_transition_matrix',
    'estimate_transition_matrices',
    'estimate',
    'get_unbiased_user_free_energies']

cdef extern from "_tram.h":
    void _tram_init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
    void _tram_update_lagrangian_mult(
        double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,  int* state_counts,
        int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
    void _tram_update_biased_conf_energies(
        double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_conf_states, double *scratch_T, double *new_biased_conf_energies)
    void _tram_get_conf_energies(
        double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
        int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
    void _tram_get_therm_energies(
        double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies)
    void _tram_normalize(
        double *conf_energies, double *biased_conf_energies, double *therm_energies,
        int n_therm_states, int n_conf_states, double *scratch_M)
    void _tram_estimate_transition_matrix(
        double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
        int n_conf_states, double *scratch_M, double *transition_matrix)
    double _tram_log_likelihood_lower_bound(
        double *old_log_lagrangian_mult, double *new_log_lagrangian_mult,
        double *old_biased_conf_energies, double *new_biased_conf_energies,
        int *count_matrices, int *state_counts,
        int n_therm_states, int n_conf_states,
        double *bias_energy_sequence, int *state_sequence, int seq_length,
        double *scratch_T, double *scratch_M, double *scratch_TM, double *scratch_MM)
    void _tram_get_log_Ref_K_i(double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,
        int *state_counts, int n_therm_states, int n_conf_states, double *scratch_M,
        double *log_R_K_i)
    void _tram_get_pointwise_unbiased_free_energies(
        int k, double *bias_energy_sequence, double *therm_energies, int *state_sequence,
        int seq_length, double *log_R_K_i, int n_therm_states, int n_conf_states,
        double *scratch_T, double *pointwise_unbiased_free_energies)


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
    _tram_init_lagrangian_mult(
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
    _tram_update_lagrangian_mult(
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
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"] 
    state_sequences, # _np.ndarray[int, ndim=1, mode="c"]
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
    bias_energy_sequences : list of numpy.ndarray(shape=(T, X_i), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequences : list of numpy.ndarray(shape=(X_i,), dtype=numpy.intc)
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
    new_biased_conf_energies[:] = _np.inf
    get_log_Ref_K_i(log_lagrangian_mult, biased_conf_energies, 
                  count_matrices, state_counts, scratch_M, log_R_K_i)
    for i in range(len(bias_energy_sequences)):
        _tram_update_biased_conf_energies(
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            <int*> _np.PyArray_DATA(state_sequences[i]),
            state_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(log_R_K_i),
            log_lagrangian_mult.shape[0],
            log_lagrangian_mult.shape[1],
            <double*> _np.PyArray_DATA(scratch_T),
            <double*> _np.PyArray_DATA(new_biased_conf_energies))

def get_log_Ref_K_i(
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=2, mode="c"] log_R_K_i not None):
    r"""
    Computes the sum of TRAM log pseudo-counts and biased_conf_energies.

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
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations
    log_R_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for sum of TRAM log pseudo-counts and biased_conf_energies
    """
    _tram_get_log_Ref_K_i(
        <double*> _np.PyArray_DATA(log_lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        log_lagrangian_mult.shape[0],
        log_lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(log_R_K_i))

def get_conf_energies(
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    state_sequences, # _np.ndarray[int, ndim=1, mode="c"] 
    _np.ndarray[double, ndim=2, mode="c"] log_R_K_i not None,
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
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations

    Returns
    -------
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        unbiased (Markov) free energies
    """
    conf_energies = _np.zeros(shape=(log_R_K_i.shape[1],), dtype=_np.float64)
    conf_energies[:] = _np.inf
    for i in range(len(bias_energy_sequences)):
        _tram_get_conf_energies(
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            <int*> _np.PyArray_DATA(state_sequences[i]),
            state_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(log_R_K_i),
            log_R_K_i.shape[0],
            log_R_K_i.shape[1],
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
    _tram_get_therm_energies(
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
    _tram_normalize(
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(biased_conf_energies),
        <double*> _np.PyArray_DATA(therm_energies),
        biased_conf_energies.shape[0],
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M))

def get_pointwise_unbiased_free_energies(
    k,
    _np.ndarray[double, ndim=2, mode="c"] log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    state_sequences, # _np.ndarray[int, ndim=1, mode="c"]
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M,    
    _np.ndarray[double, ndim=1, mode="c"] scratch_T,
    pointwise_unbiased_free_energies): # _np.ndarray[double, ndim=1, mode="c"] 
    r'''
    Compute the pointwise free energies :math:`\mu^{k}(x)` for all x.

    Parameters
    ----------
    k : int or None
        thermodynamic state, if k is None, compute pointwise free energies
        of the unbiased state.
    log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.intc)
        reduced thermodynamic free energies
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    bias_energy_sequences : list of numpy.ndarray(shape=(T, X_i), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequences : list of numpy.ndarray(shape=(X_i,), dtype=numpy.intc)
        Markov state indices for all X samples
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        number of visits to thermodynamic state K and Markov state i
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array for logsumexp operations
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations
    pointwise_unbiased_free_energies : list of numpy.ndarray(shape=(X_i), dtype=numpy.float64)
        target arrays for the pointwise free energies
    '''

    log_R_K_i = _np.zeros(shape=(state_counts.shape[0],state_counts.shape[1]), dtype=_np.float64)
    if scratch_T is None:
        scratch_T = _np.zeros(shape=(state_counts.shape[0]), dtype=_np.float64)
    if scratch_M is None:
        scratch_M = _np.zeros(shape=(state_counts.shape[1]), dtype=_np.float64)

    get_log_Ref_K_i(log_lagrangian_mult, biased_conf_energies,
                  count_matrices, state_counts, scratch_M, log_R_K_i)

    if k is None:
        k = -1

    # TODO: basic size checks for pointwise free energies!
    for i in range(len(bias_energy_sequences)):
        _tram_get_pointwise_unbiased_free_energies(
            k,
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            <double*> _np.PyArray_DATA(therm_energies),
            <int*> _np.PyArray_DATA(state_sequences[i]),
            state_sequences[i].shape[0], 
            <double*> _np.PyArray_DATA(log_R_K_i),
            log_R_K_i.shape[0],
            log_R_K_i.shape[1],
            <double*> _np.PyArray_DATA(scratch_T),
            <double*> _np.PyArray_DATA(pointwise_unbiased_free_energies[i]))

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
    _tram_estimate_transition_matrix(
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(log_lagrangian_mult[therm_state, :])),
        <double*> _np.PyArray_DATA(_np.ascontiguousarray(biased_conf_energies[therm_state, :])),
        <int*> _np.PyArray_DATA(_np.ascontiguousarray(count_matrices[therm_state, :, :])),
        biased_conf_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(transition_matrix))
    return transition_matrix

def log_likelihood_lower_bound(
    _np.ndarray[double, ndim=2, mode="c"] old_log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] new_log_lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] old_biased_conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_energies not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    state_sequences, # _np.ndarray[int, ndim=1, mode="c"]
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_MM not None):
    r"""
    Computes a lower bound on the TRAM log-likelihood with an arbitrary
    mu that is implicitly given by old_log_lagrangian_mult and
    old_biased_conf_energies.

    Parameters
    ----------
    old_log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of (old) the Lagrangian multipliers that parametrize mu
    new_log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        log of the Lagrangian multipliers that parametrize the correct
        estimate of the transtition matrix which in turn fulfills detailled
        balance wrt. new_biased_conf_energies
    old_biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        (old) reduced free energies that parametrize mu
    new_biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        (new) reduced free energies that normalize mu
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    bias_energy_sequences : list of numpy.ndarray(shape=(T, X_i), dtype=numpy.intc)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequences : list of numpy.ndarray(shape=(X_i,), dtype=numpy.intc)
        Markov state indices for all X samples
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        number of visits to thermodynamic state K and Markov state i
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    scratch_TM : numpy.ndarray(shape=(T,M), dtype=numpy.float64)
        scratch array
    scratch_MM : numpy.ndarray(shape=(M,M), dtype=numpy.float64)
        scratch array

    Note
    ----
    mu_i^{(k)} are regarded as functions of old_log_lagrangian_mult,
    old_biased_conf_energies, count_matrices, bias_energy_sequence,
    state_sequence and state_counts. By itself that doesn't have to be
    normalized. exp(-new_biased_conf_energies) are assumed to be the
    correct normalization constants for mu_i^{(k)}.

    The lower bound on the log-likelihood is computed by first projecting
    the transition matrix to close feasible point and then computing
    the (exact) log-likelihood for the combination of mu_i^{(k)},
    new_biased_conf_energies and this projected transition matrix.
    Because the projection of the transition matrix is not optimal,
    this yields only a lower bound on the true log-likelihood.
    """

    logL = _tram_log_likelihood_lower_bound(
        <double*> _np.PyArray_DATA(old_log_lagrangian_mult),
        <double*> _np.PyArray_DATA(new_log_lagrangian_mult),
        <double*> _np.PyArray_DATA(old_biased_conf_energies),
        <double*> _np.PyArray_DATA(new_biased_conf_energies),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        state_counts.shape[0],
        state_counts.shape[1],
        <double*> _np.PyArray_DATA(bias_energy_sequences[0]), # fixme
        <int*> _np.PyArray_DATA(state_sequences[0]), # fixme
        state_sequences[0].shape[0], # fixme
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_TM),
        <double*> _np.PyArray_DATA(scratch_MM))

    return logL

def estimate(count_matrices, state_counts, bias_energy_sequence, state_sequence, maxiter=1000, maxerr=1.0E-8,
             biased_conf_energies=None, log_lagrangian_mult=None, save_convergence_info=0, callback=None, N_dtram_accelerations=0):
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
    save_convergence_info : int, optional
        every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood

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
    increments : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    loglikelihoods : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of loglikelihoods

    Termination
    -----------
    The self-consitent iteration terminates when
    .. math:: max\{\max_{i,k}{\Delta \pi_i^k}, \max_k \Delta f^k \}<\mathrm{maxerr}.
    Different termination criteria can be implemented with the callback
    function. Raising `CallbackInterrupt` in the callback will cleanly
    terminate the iteration.
    """
    if biased_conf_energies is None:
        biased_conf_energies = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    if log_lagrangian_mult is None:
        log_lagrangian_mult = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
        init_lagrangian_mult(count_matrices, log_lagrangian_mult)

    increments = []
    loglikelihoods = []
    sci_count = 0

    log_R_K_i = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    scratch_T = _np.zeros(shape=(count_matrices.shape[0],), dtype=_np.float64)
    scratch_M = _np.zeros(shape=(count_matrices.shape[1],), dtype=_np.float64)
    scratch_TM = _np.zeros(shape=count_matrices.shape[0:2], dtype=_np.float64)
    scratch_MM = _np.zeros(shape=count_matrices.shape[1:3], dtype=_np.float64)
    old_biased_conf_energies = biased_conf_energies.copy()
    old_log_lagrangian_mult = log_lagrangian_mult.copy()
    old_stat_vectors = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    old_therm_energies = _np.zeros(shape=count_matrices.shape[0], dtype=_np.float64)

    for _m in range(maxiter):
        sci_count += 1 
        update_lagrangian_mult(old_log_lagrangian_mult, biased_conf_energies, count_matrices, state_counts, scratch_M, log_lagrangian_mult)
        update_biased_conf_energies(log_lagrangian_mult, old_biased_conf_energies, count_matrices, bias_energy_sequence, state_sequence,
            state_counts, log_R_K_i, scratch_M, scratch_T, biased_conf_energies)

        therm_energies = get_therm_energies(biased_conf_energies, scratch_M)
        stat_vectors = _np.exp(-biased_conf_energies + therm_energies[:, _np.newaxis])
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        delta_stat_vectors =  _np.abs(stat_vectors - old_stat_vectors)
        err = max(_np.max(delta_therm_energies),_np.max(delta_stat_vectors))

        if sci_count == save_convergence_info:
            sci_count = 0
            increments.append(err)
            loglikelihoods.append(
                log_likelihood_lower_bound(
                   log_lagrangian_mult, log_lagrangian_mult,
                   old_biased_conf_energies, biased_conf_energies,
                   count_matrices, bias_energy_sequence, state_sequence,
                   state_counts, scratch_M, scratch_T, scratch_TM, scratch_MM))

        if callback is not None:
            try:
                callback(biased_conf_energies=biased_conf_energies,
                         log_lagrangian_mult=log_lagrangian_mult,
                         therm_energies=therm_energies,
                         stat_vectors=stat_vectors,
                         old_biased_conf_energies=old_biased_conf_energies,
                         old_log_lagrangian_mult=old_log_lagrangian_mult,
                         old_stat_vectors=old_stat_vectors,
                         old_therm_energies=old_therm_energies,
                         iteration_step=_m,
                         err=err,
                         maxerr=maxerr,
                         maxiter=maxiter)
            except CallbackInterrupt:
                break

        if err < maxerr:
            break

        shift = _np.min(biased_conf_energies)
        biased_conf_energies -= shift
        old_biased_conf_energies[:] = biased_conf_energies
        old_log_lagrangian_mult[:] = log_lagrangian_mult[:]
        old_therm_energies[:] = therm_energies[:]-shift
        old_stat_vectors[:] = stat_vectors[:]

    conf_energies = get_conf_energies(bias_energy_sequence, state_sequence, log_R_K_i, scratch_T)
    therm_energies = get_therm_energies(biased_conf_energies, scratch_M)
    normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)

    if save_convergence_info == 0:
        increments = None
        loglikelihoods = None
    else:
        increments = _np.array(increments, dtype=_np.float64)
        loglikelihoods = _np.array(loglikelihoods, dtype=_np.float64)

    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, increments, loglikelihoods

def simple_error(callback=None):
    r"""
    Stopping condition for `estimate`. Can be given as the value of `callback`.
    
    Stop the estimation when the difference of the biased energies
    (logarithms of the joint probability of conformational state and
    thermodynamic state) between two iterations is smaller than `maxerr`.

    Parameters
    ----------
    callback : optional
        user call back. Because `simple_error` takes the `callback`
        slot of `estimate`, this allows to chain `simple_error` with
        another call back.
    """
    def function(**kwargs):
        biased_conf_energies = kwargs['biased_conf_energies']
        old_biased_conf_energies = kwargs['old_biased_conf_energies']
        maxerr = kwargs['maxerr']
        if callback is not None:
            callback(**kwargs)
        if _np.max(_np.abs(biased_conf_energies - old_biased_conf_energies)) < maxerr:
            raise CallbackInterrupt('biased configuration energies have converged')
    return function
