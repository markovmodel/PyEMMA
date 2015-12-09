import numpy as _np
cimport numpy as _np
import sys
from thermotools import tram as _tram
from .callback import CallbackInterrupt

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
             biased_conf_energies=None, log_lagrangian_mult=None, err_out=0, lll_out=0, callback=None,
             N_dtram_accelerations=0):
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
    err_out : int, optional
        every err_out iteration steps, store the actual increment
    lll_out : int, optional
        every lll_out iteration steps, store the actual loglikelihood

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
    err : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    lll : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of loglikelihoods        
    """
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
    bias_weight_sequence = _np.exp(-bias_energy_sequence) # TODO: shift

    err_traj = []
    lll_traj = []
    err_count = 0
    lll_count = 0

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
    #max_indices = _np.max([len(_np.where(state_counts[:,i]>0)[0]) for i in range(n_conf_states)])+1
    #indices = _np.zeros((n_conf_states, max_indices), dtype=_np.intc)
    #for i in range(n_conf_states):
    #    tmp = _np.where(state_counts[:,i]>0)[0]
    #    indices[i,0:len(tmp)] = tmp
    #    indices[i,len(tmp)] = -1
    # indices for the full matrix
    full_indices = _np.zeros((n_conf_states, n_therm_states+1), dtype=_np.intc)
    for i in range(n_conf_states):
        full_indices[i,0:n_therm_states] = range(n_therm_states)
        full_indices[i,n_therm_states] = -1

    old_biased_conf_weights = biased_conf_weights.copy()
    old_lagrangian_mult = lagrangian_mult.copy()
    old_biased_conf_energies = biased_conf_energies.copy()
    old_stat_vectors = _np.zeros(shape=state_counts.shape, dtype=_np.float64)
    old_therm_energies = _np.zeros(shape=count_matrices.shape[0], dtype=_np.float64)

    for _m in range(maxiter):
        err_count += 1
        lll_count += 1
        update_lagrangian_mult(old_lagrangian_mult, biased_conf_weights, count_matrices, state_counts, _m, lagrangian_mult)
        update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                                   state_counts, full_indices, R_K_i, scratch_TM, biased_conf_weights)

        for _n  in range(N_dtram_accelerations):
            old_biased_conf_weights[:] = biased_conf_weights[:]
            dtram_like_update(lagrangian_mult, old_biased_conf_weights, count_matrices,
                              state_counts, scratch_M, scratch_M_int, biased_conf_weights)

        partition_funcs = biased_conf_weights.sum(axis=1)
        stat_vectors = biased_conf_weights / partition_funcs[:, _np.newaxis]
        therm_energies = -_np.log(partition_funcs)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        delta_stat_vectors =  _np.abs(stat_vectors - old_stat_vectors)
        err = max(_np.max(delta_therm_energies),_np.max(delta_stat_vectors[occupied]))

        if err_count == err_out:
            err_count = 0
            err_traj.append(err)

        if lll_count == lll_out:
            lll_count = 0
            log_lagrangian_mult = _np.log(lagrangian_mult)
            old_biased_conf_energies = -_np.log(old_biased_conf_weights)
            biased_conf_energies = -_np.log(biased_conf_weights)
            logL = _tram.log_likelihood_lower_bound(log_lagrangian_mult, log_lagrangian_mult,
                                                    old_biased_conf_energies, biased_conf_energies,
                                                    count_matrices, bias_energy_sequence, state_sequence,
                                                    state_counts, scratch_M, scratch_T, scratch_TM, scratch_MM)
            lll_traj.append(logL)

        if callback is not None:
            try:
                callback(iteration_step = _m,
                         biased_conf_weights = biased_conf_weights,
                         lagrangian_mult = lagrangian_mult,
                         old_biased_conf_weights = old_biased_conf_weights,
                         old_lagrangian_mult = old_lagrangian_mult,
                         occupied = occupied,
                         err = err,
                         maxerr = maxerr,
                         maxiter = maxiter)
            except CallbackInterrupt:
                break

        if err < maxerr:
            break

        normalization_factor = _np.max(biased_conf_weights)
        biased_conf_weights /= normalization_factor
        old_lagrangian_mult[:] = lagrangian_mult[:]
        old_biased_conf_weights[:] = biased_conf_weights[:]
        old_therm_energies[:] = therm_energies[:] + _np.log(normalization_factor)
        old_stat_vectors[:] = stat_vectors[:]

    # do one dense calcualtion to find free energies of unvisited states
    update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                               state_counts, full_indices, R_K_i, scratch_TM, biased_conf_weights)

    biased_conf_energies = -_np.log(biased_conf_weights)
    log_lagrangian_mult = _np.log(lagrangian_mult)
    log_R_K_i = _np.log(R_K_i)

    conf_energies = _tram.get_conf_energies(bias_energy_sequence, state_sequence, log_R_K_i, scratch_M, scratch_T)
    therm_energies = _tram.get_therm_energies(biased_conf_energies, scratch_M)
    _tram.normalize(conf_energies, biased_conf_energies, therm_energies, scratch_M)

    if err_out == 0:
        err_traj = None
    else:
        err_traj = _np.array(err_traj, dtype=_np.float64)
    if lll_out == 0:
        lll_traj = None
    else:
        lll_traj = _np.array(lll_traj, dtype=_np.float64)

    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, err_traj, lll_traj


