import numpy as _np
import scipy
cimport numpy as _np
import ctypes
from libc.stdlib cimport free
from libc.string cimport memcpy
import msmtools
import sys

__all__ = [
    'estimate']

cdef extern from "_tram_direct.h":
    cdef struct my_sparse:
        int *rows
        int *cols
        int length
    void _update_lagrangian_mult(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
        int n_therm_states, int n_conf_states, double *new_lagrangian_mult)
    my_sparse _update_biased_conf_weights(
        double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
        int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
        int n_therm_states, int n_conf_states, int check_overlap, double *new_biased_conf_weights)

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
    check_overlap,
    _np.ndarray[double, ndim=2, mode="c"] new_biased_conf_weights not None):
        
    cdef my_sparse s
    s = _update_biased_conf_weights(
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
        int(check_overlap),
        <double*> _np.PyArray_DATA(new_biased_conf_weights))

    cdef int[:] rows
    cdef int[:] cols
    if check_overlap:
        # convert overlap info from C-code to scipy sparse
        print 'length info:', s.length
        rows = _np.zeros(s.length, dtype=_np.intc)
        cols = _np.zeros(s.length, dtype=_np.intc)
        sizof = _np.iinfo(_np.intc).bits/8
        for i in range(s.length):
            rows[i] = s.rows[i]
            cols[i] = s.cols[i]
        #memcpy(&rows[0], &(s.rows[0]), s.length*sizof)
        #memcpy(&cols[0], &(s.cols[0]), s.length*sizof)
        free(s.rows)
        free(s.cols)
        n_therm_states = lagrangian_mult.shape[0]
        n_conf_states = lagrangian_mult.shape[1]
        n_states = n_conf_states * n_therm_states
        A = scipy.sparse.coo_matrix((_np.ones_like(rows), (rows, cols)),
                                    shape=(n_states,n_states))
        # add transition links
        for k in range(n_therm_states):
            C_s = scipy.sparse.coo_matrix(count_matrices[k,:,:])
            A += scipy.sparse.coo_matrix((_np.ones_like(C_s.col),
                                          (C_s.row + k*n_conf_states, 
                                           C_s.col + k*n_conf_states)),
                                         shape=(n_states,n_states))
        indices = msmtools.estimation.largest_connected_set(A, directed=False)
        # assert np.unravel_index(2*n_conf_states+3, (n_therm_states,n_conf_states), order='C')==(2,3)
        #print A.toarray()
        # TODO: every state that has samples should be in the network!
        # because only the corresponding Z must be defined for expects..
        # indices ==_np.where(_np.ravel(state_counts)>0)[0]
        ks,iis = _np.unravel_index(indices, (n_therm_states,n_conf_states), order='C')
        if(len(_np.unique(ks))<n_therm_states or len(_np.unique(iis))<n_conf_states):
            print >> sys.stderr, "Network has fallen apart"
        else:
            print "Network is fine"

def estimate(count_matrices, state_counts, bias_energy_sequence, state_sequence, maxiter=1000, maxerr=1.0E-8, biased_conf_energies=None, log_lagrangian_mult=None):
    import sys
    print >> sys.stderr, 'Hello direct space.'
    n_therm_states = count_matrices.shape[0]
    n_conf_states = count_matrices.shape[1]
    
    assert(_np.all(state_counts == _np.maximum(count_matrices.sum(axis=1),count_matrices.sum(axis=2))))
    
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
        for _mm in range(1):
            update_lagrangian_mult(old_lagrangian_mult, biased_conf_weights, count_matrices, state_counts, lagrangian_mult)
            update_biased_conf_weights(lagrangian_mult, old_biased_conf_weights, count_matrices, bias_weight_sequence, state_sequence,
                                       state_counts, R_K_i, _m%100==0, biased_conf_weights)
            biased_conf_energies = -_np.log(biased_conf_weights)
        if _m%100==0:
            Z_K = biased_conf_weights.sum(axis=1)
            print 'Z_K', Z_K
            #print 'info (Z_K_i):', biased_conf_weights
            print 'z error:', _np.max(_np.abs(biased_conf_weights - old_biased_conf_weights))
            nz = _np.where(_np.logical_not(_np.logical_or(_np.isinf(biased_conf_energies),_np.isinf(old_biased_conf_energies))))
            print 'f error:', _np.max(_np.abs(biased_conf_energies[nz] - old_biased_conf_energies[nz]))
        if _np.max(_np.abs(biased_conf_weights - old_biased_conf_weights)) < maxerr:
            break        
        old_lagrangian_mult[:] = lagrangian_mult[:]        
        old_biased_conf_weights[:] = biased_conf_weights[:]
        # TODO shift
        old_biased_conf_energies[:] = biased_conf_energies[:]

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
    

    
    
    
