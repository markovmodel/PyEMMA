import numpy as _np
import scipy
cimport numpy as _np
import ctypes
from libc.stdlib cimport free
from libc.string cimport memcpy
import msmtools
import sys
from thermotools import tram

__all__ = [
    'estimate']

cdef extern from "_tram_direct.h":
    void _update_lagrangian_mult(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
        int n_therm_states, int n_conf_states, int iteration, double *new_lagrangian_mult)
    void _update_biased_conf_weights(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
        int *state_sequence, int *state_counts, int *indices, int indices_length, int seq_length, double *R_K_i,
        int n_therm_states, int n_conf_states, double *scratch_TM, double *new_biased_conf_weights)
    void _dtram_like_update(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int *state_counts, 
        int n_therm_states, int n_conf_states, double *scratch_M, int *scratch_M_int, double *new_biased_conf_weights)

def update_lagrangian_mult(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    iteration,
    _np.ndarray[double, ndim=2, mode="c"] new_lagrangian_mult not None):

    _update_lagrangian_mult(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        iteration,
        <double*> _np.PyArray_DATA(new_lagrangian_mult))

def update_biased_conf_weights(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_weight_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[int, ndim=2, mode="c"] indices not None,
    _np.ndarray[double, ndim=2, mode="c"] R_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] scratch_TM not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None):

    _update_biased_conf_weights(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <double*> _np.PyArray_DATA(bias_weight_sequence),
        <int*> _np.PyArray_DATA(state_sequence),
        <int*> _np.PyArray_DATA(state_counts),
        <int*> _np.PyArray_DATA(indices),
        indices.shape[1],
        state_sequence.shape[0],
        <double*> _np.PyArray_DATA(R_K_i),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_TM),
        <double*> _np.PyArray_DATA(new_biased_conf_weights))

def dtram_like_update(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[int, ndim=1, mode="c"] scratch_M_int not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None):

    _dtram_like_update(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <int*> _np.PyArray_DATA(scratch_M_int),
        <double*> _np.PyArray_DATA(new_biased_conf_weights))

def estimate(count_matrices, state_counts, bias_energy_sequence, state_sequence, maxiter=1000, maxerr=1.0E-8, 
             biased_conf_energies=None, log_lagrangian_mult=None, call_back=None, N_dtram_accelerations=0):
    import sys
    print >> sys.stderr, 'Hello direct space.'
    n_therm_states = count_matrices.shape[0]
    n_conf_states = count_matrices.shape[1]

    assert(_np.all(state_counts >= _np.maximum(count_matrices.sum(axis=1),count_matrices.sum(axis=2))))

    # init lagrangian multipliers
    if log_lagrangian_mult is None:
        lagrangian_mult = 0.5 * (count_matrices + _np.transpose(count_matrices, axes=(0,2,1))).sum(axis=2).astype(_np.float64)
    else:
        lagrangian_mult = _np.exp(log_lagrangian_mult)

    # init weights
    if biased_conf_energies is not None:
        biased_conf_weights = _np.exp(-biased_conf_energies)
    else:
        biased_conf_weights = _np.ones(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
        biased_conf_energies = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)

    # init Boltzmann factors
    bias_weight_sequence = _np.exp(-bias_energy_sequence)

    assert _np.all(state_counts >= _np.maximum(count_matrices.sum(axis=1), count_matrices.sum(axis=2)))

    R_K_i = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    scratch_TM = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    scratch_MM = _np.zeros(shape=(n_conf_states, n_conf_states), dtype=_np.float64)
    scratch_T = _np.zeros(shape=n_therm_states, dtype=_np.float64)
    scratch_M = _np.zeros(shape=n_conf_states, dtype=_np.float64)
    scratch_M_int = _np.zeros(shape=n_conf_states, dtype=_np.intc)

    occupied = _np.where(state_counts>0)

    if _np.any(_np.isinf(biased_conf_energies[occupied])):
        print >>sys.stderr, 'Warning: detected inf in biased_conf_energies.'

    # init sparse indices
    max_indices = _np.max([len(_np.where(state_counts[:,i]>0)[0]) for i in range(n_conf_states)])+1
    indices = _np.zeros((n_conf_states, max_indices), dtype=_np.intc)
    for i in range(n_conf_states):
        tmp = _np.where(state_counts[:,i]>0)[0]
        indices[i,0:len(tmp)] = tmp
        indices[i,len(tmp)] = -1
    # indices for the full matrix
    full_indices = _np.zeros((n_conf_states, n_therm_states+1), dtype=_np.intc)
    for i in range(n_conf_states):
        full_indices[i,0:n_therm_states] = range(n_therm_states)
        full_indices[i,n_therm_states] = -1


    old_biased_conf_weights = biased_conf_weights.copy()
    old_lagrangian_mult = lagrangian_mult.copy()
    old_biased_conf_energies = biased_conf_energies.copy()
    for _m in range(maxiter):
        update_lagrangian_mult(old_lagrangian_mult, biased_conf_weights, count_matrices, state_counts, _m, lagrangian_mult)
        if _m%100==0 or _m%100==99:
            update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                                       state_counts, full_indices, R_K_i, scratch_TM, biased_conf_weights)
        else:
            update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                                       state_counts, indices, R_K_i, scratch_TM, biased_conf_weights)

        for _n  in range(N_dtram_accelerations):
            old_biased_conf_weights[:] = biased_conf_weights[:]
            dtram_like_update(lagrangian_mult, old_biased_conf_weights, count_matrices,
                              state_counts, scratch_M, scratch_M_int, biased_conf_weights)

        if _m%100==99:
            biased_conf_energies = -_np.log(biased_conf_weights)
            if _np.any(_np.isinf(biased_conf_energies[occupied])):
                print >>sys.stderr, 'Warning: detected inf in biased_conf_energies.'
            old_biased_conf_energies[:] = biased_conf_energies[:]
        if _m%100==0:
            biased_conf_energies = -_np.log(biased_conf_weights)
            log_lagrangian_mult = _np.log(lagrangian_mult)
            error = _np.max(_np.abs(biased_conf_weights[occupied] - old_biased_conf_weights[occupied]))
            rel_error = _np.max(_np.abs(biased_conf_weights[occupied] - old_biased_conf_weights[occupied])/biased_conf_weights[occupied])
            log_likelihood = tram.log_likelihood_lower_bound(log_lagrangian_mult, log_lagrangian_mult,
                                                             old_biased_conf_energies, biased_conf_energies,
                                                             count_matrices, bias_energy_sequence, state_sequence,
                                                             state_counts, scratch_M, scratch_T, scratch_TM, scratch_MM)
            if call_back is not None:
                call_back(iteration=_m, old_log_lagrangian_mult=_np.log(old_lagrangian_mult), log_lagrangian_mult=log_lagrangian_mult,
                          old_biased_conf_energies=old_biased_conf_energies, biased_conf_energies=biased_conf_energies,
                          log_likelihood=log_likelihood, error=error, rel_error=rel_error)

        if _np.max(_np.abs(biased_conf_weights[occupied] - old_biased_conf_weights[occupied])/biased_conf_weights[occupied]) < maxerr:
            break

        biased_conf_weights /= _np.max(biased_conf_weights)
        old_lagrangian_mult[:] = lagrangian_mult[:]        
        old_biased_conf_weights[:] = biased_conf_weights[:]

    # do one dense calcualtion to find free energies of unvisited states
    update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                               state_counts, full_indices, R_K_i, scratch_TM, biased_conf_weights)

    biased_conf_energies = -_np.log(biased_conf_weights)
    log_lagrangian_mult = _np.log(lagrangian_mult)
    log_R_K_i = _np.log(R_K_i)

    conf_energies = tram.get_conf_energies(bias_energy_sequence, state_sequence, log_R_K_i, scratch_M, scratch_T)
    therm_energies = tram.get_therm_energies(biased_conf_energies, scratch_M)
    tram.normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)
    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult

