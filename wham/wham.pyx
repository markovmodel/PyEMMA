#
#   Copyright 2015 Christoph Wehmeyer
#

import numpy as np
cimport numpy as np

cdef extern from "_wham.h":
    void _wham_fi(
        double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
        int n_therm_states, int n_markov_states, double *scratch_T, double *f_i)
    void _wham_fk(
        double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
        double *scratch_M, double *f_K)
    void _wham_normalize(double *f_i, int n_markov_states, double *scratch_M)

def wham_fi(
    np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    np.ndarray[double, ndim=1, mode="c"] log_N_i not None,
    np.ndarray[double, ndim=1, mode="c"] f_K not None,
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    np.ndarray[double, ndim=1, mode="c"] f_i not None):
    r"""
    Calculate the reduced free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    log_N_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        log of the state counts in each of the M markov states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    _wham_fi(
        <double*> np.PyArray_DATA(log_N_K),
        <double*> np.PyArray_DATA(log_N_i),
        <double*> np.PyArray_DATA(f_K),
        <double*> np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_T),
        <double*> np.PyArray_DATA(f_i))

def wham_fk(
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Calculate the reduced thermodynamic free energies f_K
        
    Parameters
    ----------
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M markov states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    _wham_fk(
        <double*> np.PyArray_DATA(f_i),
        <double*> np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> np.PyArray_DATA(scratch_M),
        <double*> np.PyArray_DATA(f_K))

def wham_normalize(
    np.ndarray[double, ndim=1, mode="c"] f_i not None,
    np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Apply shift on the reduced free energies f_i such that the
    stationary distribution is normalized
        
    Parameters
    ----------
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M markov states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    """
    _wham_normalize(
        <double*> np.PyArray_DATA(f_i),
        f_i.shape[0],
        <double*> np.PyArray_DATA(scratch_M))
