import numpy as _np
cimport numpy as _np

__all__ = [
    'estimate']

cdef extern from "_tram_direct.h":
    void _update_lagrangian_mult(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
        int n_therm_states, int n_conf_states, double *new_lagrangian_mult)
    void _update_biased_conf_weights(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
        int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
        int n_therm_states, int n_conf_states, double *new_biased_conf_weights)

def update_lagrangian_mult(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] new_lagrangian_mult not None):

    _update_lagrangian_mult(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <int*> _np.PyArray_DATA(state_counts),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(new_lagrangian_mult))

def update_biased_conf_weights(
    _np.ndarray[double, ndim=2, mode="c"] lagrangian_mult not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_weights not None,
    _np.ndarray[int, ndim=3, mode="c"] count_matrices not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_weight_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] state_sequence not None,
    _np.ndarray[int, ndim=2, mode="c"] state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] R_K_i not None,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None):

    _update_biased_conf_weights(
        <double*> _np.PyArray_DATA(lagrangian_mult),
        <double*> _np.PyArray_DATA(biased_conf_weights),
        <int*> _np.PyArray_DATA(count_matrices),
        <double*> _np.PyArray_DATA(bias_weight_sequence),
        <int*> _np.PyArray_DATA(state_sequence),
        <int*> _np.PyArray_DATA(state_counts),
        state_sequence.shape[0],
        <double*> _np.PyArray_DATA(R_K_i),
        lagrangian_mult.shape[0],
        lagrangian_mult.shape[1],
        <double*> _np.PyArray_DATA(new_biased_conf_weights))

def estimate(count_matrices, state_counts, bias_energy_sequence, state_sequence, maxiter=1000, maxerr=1.0E-8, biased_conf_energies=None, log_lagrangian_mult=None):
    n_therm_states = count_matrices.shape[0]
    n_conf_states = count_matrices.shape[1]
    
    # init lagrangian multipliers
    lagrangian_mult = 0.5 * (count_matrices + _np.transpose(count_matrices, axes=(0,2,1))).sum(axis=2).astype(_np.float64)
    
    # init weights
    biased_conf_weights = _np.ones(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    
    # init Boltzmann factors
    bias_weight_sequence = _np.exp(-bias_energy_sequence)
    
    R_K_i = _np.zeros(shape=(n_therm_states, n_conf_states), dtype=_np.float64)
    
    old_biased_conf_weights = biased_conf_weights.copy()
    old_lagrangian_mult = lagrangian_mult.copy()
    old_biased_conf_energies = _np.zeros_like(biased_conf_weights)
    for _m in range(maxiter):
        update_lagrangian_mult(old_lagrangian_mult, biased_conf_weights, count_matrices, state_counts, lagrangian_mult)
        update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                                   state_counts, R_K_i, biased_conf_weights)
        biased_conf_energies = -_np.log(biased_conf_weights)
        # TODO shift
        if _m%100==0:
            Z_K = biased_conf_weights.sum(axis=1)
            print 'Z_K', Z_K
            #print 'info (Z_K_i):', biased_conf_weights
            print 'z error:', _np.max(_np.abs(biased_conf_weights - old_biased_conf_weights))
            nz = _np.where(_np.logical_not(_np.logical_or(_np.isinf(biased_conf_energies),_np.isinf(old_biased_conf_energies))))
            print 'f error:', _np.max(_np.abs(biased_conf_energies[nz] - old_biased_conf_energies[nz]))
        if _np.max(_np.abs(biased_conf_weights - old_biased_conf_weights)) < maxerr:
            break
        old_biased_conf_weights[:] = biased_conf_weights[:]
        old_biased_conf_energies[:] = biased_conf_energies[:]
        old_lagrangian_mult[:] = lagrangian_mult[:]

    # TODO: -likelihood computation
    #       -unbiased conf energies
    #       -transition matrices

    conf_energies = -_np.log(biased_conf_weights[0,:]) # TODO: change me!
    norm = biased_conf_weights[0,:].sum()
    biased_conf_energies = -_np.log(biased_conf_weights/norm)
    therm_energies = -_np.log(biased_conf_weights.sum(axis=1)/norm)
    log_lagrangian_mult = _np.log(lagrangian_mult)
    log_L_hist = []
    return biased_conf_energies, conf_energies, therm_energies, log_lagrangian_mult, log_L_hist
    

    
    
    
