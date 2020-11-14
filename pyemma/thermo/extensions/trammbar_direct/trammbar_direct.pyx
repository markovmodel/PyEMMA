# This file is part of thermotools.
#
# Copyright 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from . import trammbar as _trammbar

from warnings import warn as _warn
from msmtools.util.exceptions import NotConvergedWarning as _NotConvergedWarning
from .trammbar import get_pointwise_unbiased_free_energies, estimate_transition_matrix, estimate_transition_matrices

from .callback import CallbackInterrupt

__all__ = [
    'estimate_transition_matrix',
    'estimate_transition_matrices',
    'estimate',
    'get_pointwise_unbiased_free_energies']

DEF TRAMMBAR = True

cdef extern from "../tram_direct/_tram_direct.h":
    void _tram_direct_update_lagrangian_mult(
        double *lagrangian_mult, double *biased_conf_weights,
        int *count_matrices, int* state_counts, int n_therm_states, int n_conf_states,
        double *new_lagrangian_mult)
    void _tram_direct_get_Ref_K_i(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices,
        int *state_counts, int n_therm_states, int n_conf_states, double *R_K_i,
        # TRAMMBAR below
        double *therm_energies, int *equilibrium_therm_state_counts,
        double overcounting_factor)
    void _tram_direct_update_biased_conf_weights(
        double *bias_sequence, int *state_sequence, int seq_length, double *R_K_i,
        int n_therm_states, int n_conf_states, double *new_biased_conf_weights)
    void _tram_direct_dtram_like_update(
        double *lagrangian_mult, double *biased_conf_weights,
        int *count_matrices, int *state_counts, int n_therm_states, int n_conf_states,
        double *scratch_M, int *scratch_M_int, double *new_biased_conf_weights)

def update_lagrangian_mult(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] new_lagrangian_mult not None):
    _tram_direct_update_lagrangian_mult(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(new_lagrangian_mult))

def get_Ref_K_i(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] R_K_i not None,
    # TRAMMBAR below
    _np.ndarray[double, ndim=1, mode="c"] therm_weights=None,
    _np.ndarray[int, ndim=1, mode="c"] equilibrium_therm_state_counts=None,
    double overcounting_factor=1.0):
    r"""
    Computes the product of TRAM pseudo-counts and biased_conf_weights.

    Parameters
    ----------
    lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        Lagrangian multipliers
    biased_conf_weights : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        exp(-reduced free energies)
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        multistate count matrix
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        number of visits to thermodynamic state K and Markov state i
    R_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        target array for product of TRAM pseudo-counts and biased_conf_weights
    """
    _tram_direct_get_Ref_K_i(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(R_K_i),
        <double*> _np.PyArray_DATA(therm_weights) if therm_weights is not None else NULL,
        <int*> _np.PyArray_DATA(equilibrium_therm_state_counts) if equilibrium_therm_state_counts is not None else NULL,
        overcounting_factor)

def update_biased_conf_weights(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    bias_weight_sequences,
    state_sequences,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] R_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None,
    # TRAMMBAR below
    _np.ndarray[double, ndim=1, mode="c"] therm_weights=None,
    equilibrium_bias_weight_sequences=None,
    equilibrium_state_sequences=None,
    _np.ndarray[int, ndim=1, mode="c"] equilibrium_therm_state_counts=None,
    double overcounting_factor=1.0):

    new_biased_conf_weights[:] = 0.0
    get_Ref_K_i(lagrangian_mult, biased_conf_weights, count_matrices,
                state_counts, R_K_i,
                therm_weights=therm_weights,
                equilibrium_therm_state_counts=equilibrium_therm_state_counts,
                overcounting_factor=overcounting_factor)
    for i in range(len(bias_weight_sequences)):
        _tram_direct_update_biased_conf_weights(
            <double*> _np.PyArray_DATA(bias_weight_sequences[i]),
            <int*> _np.PyArray_DATA(state_sequences[i]),
            state_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(R_K_i),
            lagrangian_mult.shape[0],
            lagrangian_mult.shape[1],
            <double*> _np.PyArray_DATA(new_biased_conf_weights))
    if TRAMMBAR:
        if equilibrium_bias_weight_sequences is not None:
            new_biased_conf_weights *= overcounting_factor
            for i in range(len(equilibrium_bias_weight_sequences)):
                _tram_direct_update_biased_conf_weights(
                    <double*> _np.PyArray_DATA(equilibrium_bias_weight_sequences[i]),
                    <int*> _np.PyArray_DATA(equilibrium_state_sequences[i]),
                    equilibrium_state_sequences[i].shape[0],
                    <double*> _np.PyArray_DATA(R_K_i),
                    lagrangian_mult.shape[0],
                    lagrangian_mult.shape[1],
                    <double*> _np.PyArray_DATA(new_biased_conf_weights))

def dtram_like_update(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[int, ndim=1, mode="c"] scratch_M_int not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None):
    _tram_direct_dtram_like_update(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <int*> _np.PyArray_DATA(scratch_M_int),
        <double*> _np.PyArray_DATA(new_biased_conf_weights))

def estimate(
    count_matrices, state_counts, bias_energy_sequences, state_sequences,
    maxiter=1000, maxerr=1.0E-8, save_convergence_info=0,
    biased_conf_energies=None, log_lagrangian_mult=None, callback=None,
    N_dtram_accelerations=0,
    equilibrium_therm_state_counts=None,
    equilibrium_bias_energy_sequences=None, equilibrium_state_sequences=None,
    overcounting_factor = 1.0):
    r"""
    Estimate the reduced discrete state free energies and thermodynamic free energies

    Parameters
    ----------
    count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
        transition count matrices for all T thermodynamic states
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        state counts for all M discrete and T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    state_sequences : list of numpy.ndarray(shape=(X_i), dtype=numpy.float64)
        discrete state indices for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    save_convergence_info : int, optional
        every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood
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
    increments : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    loglikelihoods : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of loglikelihoods     
    """
    if not TRAMMBAR:
        assert equilibrium_therm_state_counts is None
    
    n_therm_states = count_matrices.shape[0]
    n_conf_states = count_matrices.shape[1]
    assert len(state_sequences)==len(bias_energy_sequences)
    for s, b in zip(state_sequences, bias_energy_sequences):
        assert s.ndim == 1
        assert s.dtype == _np.intc
        assert b.ndim == 2
        assert b.dtype == _np.float64
        assert s.shape[0] == b.shape[0]
        assert b.shape[1] == n_therm_states
        assert s.flags.c_contiguous
        assert b.flags.c_contiguous
    if TRAMMBAR:
        assert N_dtram_accelerations == 0
        if equilibrium_state_sequences is not None:
            assert len(equilibrium_state_sequences) == len(equilibrium_bias_energy_sequences)
            for s, b in zip(equilibrium_state_sequences, equilibrium_bias_energy_sequences):
                assert s.ndim == 1
                assert s.dtype == _np.intc
                assert b.ndim == 2
                assert b.dtype == _np.float64
                assert s.shape[0] == b.shape[0]
                assert b.shape[1] == count_matrices.shape[0]
                assert s.flags.c_contiguous
                assert b.flags.c_contiguous
    else:
        assert equilibrium_bias_energy_sequences is None
        assert equilibrium_state_sequences is None

    assert(_np.all(
        state_counts >= _np.maximum(count_matrices.sum(axis=1), count_matrices.sum(axis=2))))

    # init lagrangian multipliers
    if log_lagrangian_mult is None:
        lagrangian_mult = 0.5 * (count_matrices + _np.transpose(
            count_matrices, axes=(0, 2, 1))).sum(axis=2).astype(_np.float64)
    else:
        lagrangian_mult = _np.exp(log_lagrangian_mult)
    # exploit invariance w.r.t. simultaneous scaling of energies and free energies
    # standard quantities-> scaled quantities
    # bias_energy^k(x)   -> bias_energy^k(x) - alpha^k
    # free_energy_i^k    -> free_energy_i^k - alpha^k
    # log \tilde{R}_i^k  -> log \tilde{R}_i^k - alpha^k
    if TRAMMBAR:
        if equilibrium_bias_energy_sequences is not None:
            all_bias_energy_sequences = bias_energy_sequences + equilibrium_bias_energy_sequences
        else:
            all_bias_energy_sequences = bias_energy_sequences
    else:
        all_bias_energy_sequences = bias_energy_sequences
    shift = _np.min([_np.min(b, axis=0) for b in all_bias_energy_sequences], axis=0) # minimum energy for every th. state        
    # init weights
    if biased_conf_energies is not None:
        biased_conf_weights = _np.exp(shift[:, _np.newaxis] - biased_conf_energies)
    else:
        biased_conf_weights = _np.ones(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
        biased_conf_energies = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    # init Boltzmann factors # TODO: offer in-place option
    bias_weight_sequences = [_np.exp(shift - b) for b in bias_energy_sequences]
    if TRAMMBAR:
        if equilibrium_bias_energy_sequences is not None:
            equilibrium_bias_weight_sequences = [_np.exp(shift - b) for b in equilibrium_bias_energy_sequences]
        else:
            equilibrium_bias_weight_sequences = None
    else:
        equilibrium_bias_weight_sequences =None
    increments = []
    loglikelihoods = []
    sci_count = 0
    R_K_i = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    scratch_TM = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    scratch_TM2 = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    scratch_MM = _np.zeros(shape=(n_conf_states, n_conf_states), dtype=_np.float64)
    scratch_T = _np.zeros(shape=n_therm_states, dtype=_np.float64)
    scratch_M = _np.zeros(shape=n_conf_states, dtype=_np.float64)
    scratch_M_int = _np.zeros(shape=n_conf_states, dtype=_np.intc)
    occupied = _np.where(state_counts > 0)
    if _np.any(_np.isinf(biased_conf_energies[occupied])):
        print >>sys.stderr, 'Warning: detected inf in biased_conf_energies.' # TODO: possible Python3 violation
    partition_funcs = biased_conf_weights.sum(axis=1)
    old_partition_funcs = partition_funcs.copy()
    old_biased_conf_weights = biased_conf_weights.copy()
    old_lagrangian_mult = lagrangian_mult.copy()
    old_biased_conf_energies = biased_conf_energies.copy()
    old_stat_vectors = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    old_therm_energies = _np.zeros(shape=count_matrices.shape[0], dtype=_np.float64)
    for m in range(maxiter):
        sci_count += 1
        update_lagrangian_mult(
            old_lagrangian_mult, biased_conf_weights, count_matrices, state_counts, lagrangian_mult)
        update_biased_conf_weights(
            lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequences,
            state_sequences, state_counts, R_K_i, biased_conf_weights,
            therm_weights=old_partition_funcs,
            equilibrium_bias_weight_sequences=equilibrium_bias_weight_sequences,
            equilibrium_state_sequences=equilibrium_state_sequences,
            equilibrium_therm_state_counts=equilibrium_therm_state_counts,
            overcounting_factor=overcounting_factor)
        for _n  in range(N_dtram_accelerations):
            old_biased_conf_weights[:] = biased_conf_weights[:]
            dtram_like_update(
                lagrangian_mult, old_biased_conf_weights, count_matrices,
                state_counts, scratch_M, scratch_M_int, biased_conf_weights)
        partition_funcs[:] = biased_conf_weights.sum(axis=1)
        stat_vectors = biased_conf_weights / partition_funcs[:, _np.newaxis]
        therm_energies = -_np.log(partition_funcs)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        delta_stat_vectors =  _np.abs(stat_vectors - old_stat_vectors)
        err = max(_np.max(delta_therm_energies), _np.max(delta_stat_vectors[occupied]))
        if sci_count == save_convergence_info:
            sci_count = 0
            increments.append(err)
            with _np.errstate(divide='ignore'):
                log_lagrangian_mult = _np.log(lagrangian_mult)
                biased_conf_energies = shift[:, _np.newaxis] - _np.log(biased_conf_weights) # can contain -inf for empty state
            logL = _trammbar.log_likelihood_lower_bound(
                log_lagrangian_mult, biased_conf_energies, count_matrices,
                bias_energy_sequences, state_sequences, state_counts,
                scratch_TM2, scratch_M, scratch_T, scratch_TM, scratch_MM,
                therm_energies=therm_energies,
                equilibrium_bias_energy_sequences=equilibrium_bias_energy_sequences,
                equilibrium_state_sequences=equilibrium_state_sequences,
                equilibrium_therm_state_counts=equilibrium_therm_state_counts,
                overcounting_factor=overcounting_factor)
            loglikelihoods.append(logL)
        if callback is not None:
            try:
                callback(iteration_step = m,
                         biased_conf_weights = biased_conf_weights,
                         lagrangian_mult = lagrangian_mult,
                         old_biased_conf_weights = old_biased_conf_weights,
                         old_lagrangian_mult = old_lagrangian_mult,
                         occupied = occupied,
                         shift = shift,
                         err = err,
                         maxerr = maxerr,
                         maxiter = maxiter)
            except CallbackInterrupt:
                break
        if err < maxerr:
            break
        else:
            normalization_factor = _np.max(biased_conf_weights)
            biased_conf_weights /= normalization_factor
            old_lagrangian_mult[:] = lagrangian_mult[:]
            old_biased_conf_weights[:] = biased_conf_weights[:]
            old_therm_energies[:] = therm_energies[:] + _np.log(normalization_factor)
            old_partition_funcs[:] = partition_funcs / normalization_factor
            old_stat_vectors[:] = stat_vectors[:]
    with _np.errstate(divide='ignore'):
        biased_conf_energies = shift[:, _np.newaxis] - _np.log(biased_conf_weights) # can contain -inf for empty states
        log_lagrangian_mult = _np.log(lagrangian_mult)
        log_R_K_i = _np.log(R_K_i) + shift[:, _np.newaxis]
    conf_energies = _trammbar.get_conf_energies(
        bias_energy_sequences, state_sequences, log_R_K_i, scratch_T,
        equilibrium_bias_energy_sequences=equilibrium_bias_energy_sequences,
        equilibrium_state_sequences=equilibrium_state_sequences,
        overcounting_factor=overcounting_factor)
    therm_energies = _trammbar.get_therm_energies(biased_conf_energies, scratch_M)
    _trammbar.normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)
    if err >= maxerr:
        _warn("TRAM did not converge: last increment = %.5e" % err, _NotConvergedWarning)
    if save_convergence_info == 0:
        increments = None
        loglikelihoods = None
    else:
        increments = _np.array(increments, dtype=_np.float64)
        loglikelihoods = _np.array(loglikelihoods, dtype=_np.float64)
    return biased_conf_energies, conf_energies, therm_energies, \
        log_lagrangian_mult, increments, loglikelihoods
